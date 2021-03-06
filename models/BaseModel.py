# -*- coding: utf-8 -*-
# Copyright (c) 2018 João Martins
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import numpy as np
import tensorflow as tf
from utils.padding import tf_pad_wrap


class BaseModel(object):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        # I like to keep the best HP found so far inside the model itself
        # This is a mechanism to load the best HP and override the configuration
        # if config.best: # FIXME: Add a way to store and load best configurations
        #     config.update(self.get_best_config(config['env_name']))

        # I make a `deepcopy` of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously over configurations
        self.config = copy.deepcopy(config)

        # if config.debug:  # This is a personal check i like to do
        for opt in config.__dict__:
            print('OPTIONS: \t{:>20}:\t\t{}'.format(opt, config.__dict__[opt]))

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        # self.random_seed = self.config.random_seed

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.result_dir = self.config.result_dir
        self.max_iter = self.config.max_iter
        self.lr = self.config.lr
        self.batch_normalization = self.config.no_batch_norm
        self.dropout_seed = self.config.seed
        if self.config.mode == 'infer':
            self.batch_size = 1
        else:
            self.batch_size = self.config.batch_size
        # self.dropout_keep_prob = self.config.dropout

        # Again, child Model should provide its own build_grap function
        self.layers = []
        self.graph = tf.Graph()
        self._set_agent_props()
        self._build_graph()

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.increment_epoch = tf.assign(self.epoch, self.epoch + 1)
            self.global_step_var = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = self.config.lr
            with tf.variable_scope('validation_average'):
                self.average_pl = tf.placeholder(tf.float32)
                self.epoch_validation = tf.summary.scalar("epoch_average_accuracy", self.average_pl,
                                                         collections=['accuracy_per_epoch'])
                self.step_validation = tf.summary.scalar("validation_average_accuracy", self.average_pl,
                                                         collections=['accuracy_per_steps'])
            # Entropy and loss for the models
            with tf.variable_scope('entropy'):
                self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[-1]['preactivation'],
                                                                       labels=self.train_labels)
                self.variable_summaries(self.entropy)
            with tf.variable_scope('regularization'):
                self.regularization = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'biases' not in v.name]) * \
                                      self.config.l2_beta
                tf.summary.scalar('regularization', self.regularization)

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(self.entropy) + self.regularization
                tf.summary.scalar('loss', self.loss)

            with tf.variable_scope('train'):
                if self.config.optimizer == 'Adam':
                        self.train_step = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                                                 epsilon=1e-08)\
                            .minimize(self.loss, global_step=self.global_step_var)
                elif self.config.optimizer == 'Nesterov':
                        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.5,
                                                                     use_nesterov=True)\
                            .minimize(self.loss, global_step=self.global_step_var)
                elif self.config.optimizer == 'AdaDelta':
                        self.train_step = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95,
                                                                     epsilon=1e-08)\
                            .minimize(self.loss, global_step=self.global_step_var)

            # if self.config.debug:
            with tf.variable_scope('train_accuracy'):
                with tf.variable_scope('correct_prediction'):
                    self.correct_prediction = tf.equal(tf.argmax(self.layers[-1]['output_layer'], 1),
                                                       tf.argmax(self.train_labels, 1))
                with tf.variable_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                self.train_accuracy = tf.summary.scalar('train_accuracy', self.accuracy, collections=['train_accuracy'])

            with tf.variable_scope('validation_accuracy'):
                with tf.variable_scope('correct_prediction'):
                    self.test_correct_prediction = tf.equal(tf.argmax(self.layers[-1]['output_layer'], 1),
                                                            tf.argmax(self.train_labels, 1))
                with tf.variable_scope('accuracy'):
                    self.test_accuracy = tf.reduce_mean(tf.cast(self.test_correct_prediction, tf.float32))
                # self.variable_summaries(self.test_accuracy)

            print("Number of parameters: ",
                  sum(reduce(lambda x, y: x * y, v.get_shape().as_list()) for v in tf.trainable_variables()))

            self.merged_summaries = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
            self.saver = tf.train.Saver(
                max_to_keep=1,
            )
            self.init_op = tf.global_variables_initializer()
            assert self.layers[-1]['preactivation'].graph is self.train_step.graph # Assure built on the same graph


        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(os.path.join(self.result_dir, 'train_summary'), self.sess.graph)
        self.test_sw = tf.summary.FileWriter(os.path.join(self.result_dir, 'test_summary'), self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

        # At the end of this function, you want your model to be ready!

    @staticmethod
    def _get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def _set_agent_props(self):
        raise Exception('The set_agent_props function must be overriden by the agent')

    def _build_graph(self):
        raise Exception('The build_graph function must be overriden by the agent')

    # def learn_from_epoch(self, grid_matrix, labels, gradient_batch_sizes):
    #     # I like to separate the function to train per epoch and the function to train globally
    #     raise Exception('The learn_from_epoch function must be overriden by the agent')

    # def eval_from_epoch(self, grid_matrix, labels, gradient_batch_sizes):
    #     # I like to separate the function to train per epoch and the function to train globally
    #     raise Exception('The eval_from_epoch function must be overriden by the agent')

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            with self.graph.as_default():
                tf.global_variables_initializer().run(session=self.sess)
        else:
            if self.config.debug:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def train(self, train_data, validation_data, save_every=1):
        # This function is usually common to all your models, Here is an example:
        starting_epoch = self.sess.run(self.epoch)
        for epoch_id in range(starting_epoch, self.max_iter):
            epoch_id = self.sess.run(self.increment_epoch)

            print('Epoch:\t{}'.format(epoch_id))
            while train_data.hold():
                indices, values, shape, hots_index = train_data.next(self.config.batch_size)
                self.learn_from_batch(indices, values, hots_index, validation_data)

            self.validate(validation_data)

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:  # FIXME: Should be 0, set at negative for no saving
                self.save(epoch_id)
            train_data.restart()

    def validate(self, validation_data, partial=False):
        errors = []
        while validation_data.hold():
            indices, values, shape, hots_index = validation_data.next(self.config.batch_size)

            batch_errors = self.eval_from_batch(indices, values, hots_index)
            errors += batch_errors
        feed_dict = dict({self.average_pl: sum(errors)/len(errors)})

        if partial:
            summary_str, global_step = self.sess.run([self.step_validation, self.global_step_var],
                                                     feed_dict=feed_dict)
            self.test_sw.add_summary(summary_str, global_step)
            print('Step {} validation accuracy: {:.2f}%'.format(global_step, (sum(errors) / len(errors)) * 100))
        else:
            summary_str, epoch, global_step = self.sess.run([self.epoch_validation, self.epoch, self.global_step_var],
                                                            feed_dict=feed_dict)
            self.test_sw.add_summary(summary_str, epoch)
            print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, (sum(errors) / len(errors)) * 100))
        validation_data.restart()

    def test(self, test_data):
        accuracy = []
        more_data = True
        while more_data:
            batch, gradient_batch_sizes = test_data.next(self.config.max_batch_size,
                                                         subbatch_max_size=self.config.subbatch_max_size,
                                                         enforce_protein_boundaries=False)
            more_data = (test_data.feature_index != 0)

            grid_matrix = batch["high_res"]
            labels = batch["model_output"]

            batch_accuracy = self.eval_from_epoch(grid_matrix, labels, gradient_batch_sizes)
            accuracy += batch_accuracy
        print('Testing accuracy: {:.2f}%'.format((np.sum(accuracy)/len(accuracy))*100))

    def save(self, epoch):
        if self.config.debug:
            print('Saving to {} on epoch {}'.format(self.result_dir, epoch))
        self.saver.save(self.sess, os.path.join(self.result_dir, 'epoch_{}'.format(epoch)))

        # I always keep the configuration that
        if not os.path.isfile(os.path.join(self.result_dir, 'config.json')):
            with open(os.path.join(self.result_dir,'config.json'), 'w') as f:
                json.dump(self.config.__dict__, f)

    def infer(self, infer_data, index):
        aa1 = "ACDEFGHIKLMNPQRSTVWYX"
        indices, values, hots = infer_data.infer(index)

        feed_dict = dict({self.indices: indices,
                          self.values: values,
                          self.dropout_keep_prob: 1.0,
                          self.phase_train: False})

        result = self.sess.run(self.layers[-1]['output_layer'], feed_dict=feed_dict)
        print('Expected:\t {}'.format(aa1[np.argmax(hots)]))
        print('Predicted:\t {}'.format(aa1[np.argmax(result[0])]))
        for index, val in enumerate(aa1):
            print('{}: {}'.format(val, result[0][index]))

    def print_layer(self, layers, idx, name):
        if name == 'W':
            size = int(np.prod(layers[-1][name].get_shape()))
        else:
            size = int(np.prod(layers[-1][name].get_shape()[1:]))
        print("layer {} (high res) - {:>15}: {} [size {:,.0f}]" \
              "".format(len(layers), name, layers[idx][name].get_shape(), size))

    def variable_summaries(self, var):
        pass
        # """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        # with tf.variable_scope('summaries'):
        #     mean = tf.reduce_mean(var)
        #     tf.summary.scalar('mean', mean)
        #     with tf.variable_scope('stddev'):
        #         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #     tf.summary.scalar('stddev', stddev)
        #     tf.summary.scalar('max', tf.reduce_max(var))
        #     tf.summary.scalar('min', tf.reduce_min(var))
        #     tf.summary.histogram('histogram', var)

    def weight_variable(self, shape, layer_name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=self.config.initial_stddev, name='weight_{}'.format(layer_name))
        return tf.Variable(initial)

    def bias_variable(self, shape, layer_name):
        """Create a bias variable with appropriate initialization."""
        # initial = tf.truncated_normal(shape, stddev=self.config.initial_stddev, name='bias_{}'.format(layer_name))
        initial = tf.zeros(shape, name='bias_{}'.format(layer_name))
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, output_dim, layer_name, dropout_keep_prob, act=tf.nn.relu, conv2fc=False,
                 batch_normalize=True):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        input_dim = np.prod(input_tensor.get_shape().as_list()[1:])
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.variable_scope(layer_name):
            if conv2fc:
                input_tensor = tf.reshape(input_tensor, (-1, np.prod(input_tensor.get_shape().as_list()[1:])))
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim], layer_name)
                self.variable_summaries(weights)
            with tf.variable_scope('biases'):
                biases = self.bias_variable([output_dim], layer_name)
                self.variable_summaries(biases)
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
                # tf.summary.histogram('pre_activations', preactivate)
            if batch_normalize:
                with tf.variable_scope('batch_norm'):
                    # layer_mean, layer_variance = tf.nn.moments(preactivate, axes=[0])
                    # preactivate = tf.nn.batch_normalization(preactivate, layer_mean, layer_variance, offset=None,
                    #                                         scale=None, variance_epsilon=1e-8)
                    # tf.summary.histogram('batch_norm', preactivate)
                    preactivate = self.batch_norm(preactivate, output_dim, layer_name, self.phase_train,
                                                  convolution=False)
            activations = act(preactivate, name='activation')
            activations = tf.nn.dropout(activations, dropout_keep_prob, seed=self.dropout_seed)
            # tf.summary.histogram('activations', activations)
            return preactivate, activations, weights

    def conv_layer(self, input_tensor, filter_size_3d, output_depth, stride, layer_name, act=tf.nn.relu, auto_pad=True,
                   explicit_pad=None, batch_normalize=True):
        if auto_pad:
            input_tensor = self.circular_pad(input_tensor, filter_size_3d, use_r_padding=False)
        elif explicit_pad:
            r_pad, theta_pad, phi_pad = explicit_pad
            input_tensor = tf.pad(input_tensor, [(0, 0), (r_pad, r_pad), (theta_pad, theta_pad),
                                      (0, 0), (0, 0)], "CONSTANT")
            input_tensor = tf_pad_wrap(input_tensor, [(0, 0), (0, 0), (0, 0), (phi_pad, phi_pad),
                                                      (0, 0)])
        with tf.variable_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights'):
                weights = self.weight_variable(filter_size_3d + input_tensor.get_shape().as_list()[-1:] +
                                               [output_depth], layer_name)
                self.variable_summaries(weights)
            with tf.variable_scope('biases'):
                biases = self.bias_variable([output_depth], layer_name)
                self.variable_summaries(biases)
            with tf.variable_scope('convolution'):
                preactivate = tf.nn.bias_add(tf.nn.conv3d(input_tensor, filter=weights, strides=stride,
                                                          padding='VALID'), biases)
                # tf.summary.histogram('pre_activations', preactivate)
            if batch_normalize:
                with tf.variable_scope('batch_norm'):
                    # layer_mean, layer_variance = tf.nn.moments(preactivate, axes=[0, 1, 2, 3])
                    # preactivate = tf.nn.batch_normalization(preactivate, layer_mean, layer_variance, offset=None,
                    #                                         scale=None, variance_epsilon=1e-8)
                    # tf.summary.histogram('batch_norm', preactivate)
                    preactivate = self.batch_norm(preactivate, output_depth, layer_name, self.phase_train,
                                                  convolution=True)
            activations = act(preactivate, name='activation')
            # tf.summary.histogram('activations', activations)
            return activations

    def pool_layer(self, input_tensor, ksize, stride, layer_name, pool_type=tf.nn.avg_pool3d, padded=True):
        with tf.variable_scope(layer_name):
            if padded:
                input_tensor = tf_pad_wrap(input_tensor,
                                           [(0, 0), (0, 0), (0, 0), (ksize[3] // 2, ksize[3] // 2), (0, 0)])
            pool = pool_type(input_tensor, ksize=ksize, strides=stride, padding='VALID',
                             name='pool_{}'.format(layer_name))
            # tf.summary.histogram('pooling', pool)
            return pool

    def circular_pad(self, input_tensor, filter_size_3d, use_r_padding=True):
        # Pad input with periodic image
        window_size_r, window_size_theta, window_size_phi = filter_size_3d
        if use_r_padding:
            r_padding = (window_size_r // 2, window_size_r // 2)
        else:
            r_padding = (0, 0)

        input = tf.pad(input_tensor, [(0, 0), r_padding, (window_size_theta // 2, window_size_theta // 2),
                                      (0, 0), (0, 0)], "CONSTANT")

        # Pad input with periodic image - only in phi
        padded_input = tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (window_size_phi // 2, window_size_phi // 2), (0, 0)])
        return padded_input

    def batch_norm(self, x, n_out, layer, phase_train, convolution=False):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        if convolution:
            with tf.variable_scope('{}_bn_convolution'.format(layer)):
                beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                   name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    name='gamma', trainable=True)
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.5)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        else:
            with tf.variable_scope('{}_bn_nn'.format(layer)):
                beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                   name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    name='gamma', trainable=True)
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.5)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def learn_from_batch(self, indices, values, hots_index, validation_data):
        feed_dict = dict({self.indices: indices,
                          self.values: values,
                          self.train_labels: hots_index,
                          self.dropout_keep_prob: self.config.dropout,
                          self.phase_train: True})
        # feed_dict = dict({self.input: (indices, values, shape),
        #                   self.train_labels: hots_index,
        #                   self.dropout_keep_prob: self.config.dropout,
        #                   self.phase_train: True})

        if self.config.debug:
            _, loss_value, summary, global_step, accuracy = self.sess.run(
                [self.train_step, self.loss, self.merged_summaries, self.global_step_var,
                 self.accuracy], feed_dict=feed_dict)
            self.sw.add_summary(summary, global_step=global_step)
            print('Loss step {}: {:.2f} [Sub-batch error: {:.0f}%]'.format(global_step, loss_value,
                                                                           100 - (100 * accuracy)))
        else:
            # _, loss_value, global_step = self.sess.run(
            #     [self.train_step, self.loss, self.global_step_var], feed_dict=feed_dict)
            _, loss_value, summary, global_step = self.sess.run(
                [self.train_step, self.loss, self.merged_summaries, self.global_step_var], feed_dict=feed_dict)
            self.sw.add_summary(summary, global_step=global_step)

            if global_step % 5000 == 0 and global_step != 0:
                accuracy, train_accuracy = self.sess.run([self.accuracy, self.train_accuracy], feed_dict=feed_dict)
                print('Loss step {}: {:.2f} [Sub-batch error: {:.0f}%]'.format(global_step, loss_value,
                                                                               100 - (100 * accuracy)))
                self.sw.add_summary(train_accuracy, global_step=global_step)
                self.validate(validation_data, partial=True)
            elif global_step % 100 == 0 and global_step != 0:
                accuracy, summary = self.sess.run([self.accuracy, self.train_accuracy], feed_dict=feed_dict)
                self.sw.add_summary(summary, global_step=global_step)
                print('Loss step {}: {:.2f} [Sub-batch error: {:.0f}%]'.format(global_step, loss_value,
                                                                               100 - (100 * accuracy)))

            else:
                print('Loss step {}: {:.2f}'.format(global_step, loss_value))

    def eval_from_batch(self, indices, values, hots_index):
        losses = []
        feed_dict = dict({self.indices: indices,
                          self.values: values,
                          self.train_labels: hots_index,
                          self.dropout_keep_prob: 1.0,
                          self.phase_train: False})

        accuracy = self.sess.run(self.test_accuracy, feed_dict=feed_dict)
        losses.append(accuracy)
        return losses
