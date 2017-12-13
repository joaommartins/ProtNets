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

        if config.debug:  # This is a personal check i like to do
            for opt in config.__dict__:
                print('DEBUG: \t{:>20}:\t\t{}'.format(opt, config.__dict__[opt]))

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config.random_seed

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.result_dir = self.config.result_dir
        self.max_iter = self.config.max_iter
        self.lr = self.config.lr
        self.dropout_keep_prob = self.config.dropout

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_agent_props()

        # Again, child Model should provide its own build_grap function
        self.layers = []
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.increment_epoch = tf.assign(self.epoch, self.epoch + 1)
            self.global_step_var = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = self.config.lr
            with tf.name_scope('validation_average'):
                self.average_pl = tf.placeholder(tf.float32)
                self.average_summary = tf.summary.scalar("epoch_average_accuracy", self.average_pl,
                                                         collections=['accuracy_per_epoch'])
            # Entropy and loss for the models
            with tf.name_scope('entropy'):
                self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob_dist, labels=self.train_labels)
                self.variable_summaries(self.entropy)
            with tf.name_scope('regularization'):
                self.regularization = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'biases' not in v.name]) * \
                                      self.config.l2_beta
                tf.summary.scalar('regularization', self.regularization)

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(self.entropy) + self.regularization
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                                         epsilon=1e-08).minimize(self.loss,
                                                                                 global_step=self.global_step_var)

            with tf.name_scope('train_accuracy'):
                with tf.name_scope('correct_prediction'):
                    self.correct_prediction = tf.equal(tf.argmax(self.layers[-1]['output_layer'], 1),
                                                       tf.argmax(self.train_labels, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
                tf.summary.histogram('accuracy', self.accuracy)

            with tf.name_scope('validation_accuracy'):
                with tf.name_scope('correct_prediction'):
                    self.test_correct_prediction = tf.equal(tf.argmax(self.layers[-1]['output_layer'], 1),
                                                            tf.argmax(self.train_labels, 1))
                with tf.name_scope('accuracy'):
                    self.test_accuracy = tf.reduce_mean(tf.cast(self.test_correct_prediction, tf.float32))
                # self.variable_summaries(self.test_accuracy)

            self.merged_summaries = tf.summary.merge_all()
            self.average_per_epoch = tf.summary.merge_all(key='accuracy_per_epoch')
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(
                max_to_keep=1,
            )

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
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def set_agent_props(self):
        raise Exception('The set_agent_props function must be overriden by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the agent')

    def infer(self):
        raise Exception('The infer function must be overriden by the agent')

    def learn_from_epoch(self, grid_matrix, labels, gradient_batch_sizes, step):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def eval_from_epoch(self, grid_matrix, labels, gradient_batch_sizes):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The eval_from_epoch function must be overriden by the agent')

    def train(self, train_data, validation_data, save_every=1):
        # This function is usually common to all your models, Here is an example:
        starting_epoch = self.sess.run(self.epoch)
        for epoch_id in range(starting_epoch, self.max_iter):
            epoch_id = self.sess.run(self.increment_epoch)
            more_data = True

            print('Epoch:\t{}'.format(epoch_id))
            while more_data:
                batch, gradient_batch_sizes = train_data.next(self.config.max_batch_size,
                                                              subbatch_max_size=self.config.subbatch_max_size,
                                                              enforce_protein_boundaries=False)
                more_data = (train_data.feature_index != 0)

                grid_matrix = batch["high_res"]
                labels = batch["model_output"]

                self.learn_from_epoch(grid_matrix, labels, gradient_batch_sizes)

            self.validate(validation_data)

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:  # FIXME: Should be 0, set at negative for no saving
                self.save(epoch_id)

    # def eval(self, validation_data):

    def validate(self, validation_data):
        errors = []
        more_data = True
        while more_data:
            batch, gradient_batch_sizes = validation_data.next(self.config.max_batch_size,
                                                               subbatch_max_size=self.config.subbatch_max_size,
                                                               enforce_protein_boundaries=False)
            more_data = (validation_data.feature_index != 0)

            grid_matrix = batch["high_res"]
            labels = batch["model_output"]
            batch_errors = self.eval_from_epoch(grid_matrix, labels, gradient_batch_sizes)
            errors += batch_errors
        feed_dict = dict({self.average_pl: sum(errors)/len(errors)})
        summary_str, epoch = self.sess.run([self.average_per_epoch, self.epoch], feed_dict=feed_dict)
        self.test_sw.add_summary(summary_str, epoch)
        print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, (sum(errors)/len(errors)) * 100))

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
        # This function is usually common to all your models, Here is an example:
        # global_step_t = tf.train.get_global_step(self.graph)
        # global_step = self.sess.run([global_step_t])
        if self.config.debug:
            print('Saving to {} on epoch {}'.format(self.result_dir, epoch))
        self.saver.save(self.sess, os.path.join(self.result_dir, 'epoch_{}'.format(epoch)))

        # I always keep the configuration that
        if not os.path.isfile(os.path.join(self.result_dir, 'config.json')):
            with open(os.path.join(self.result_dir,'config.json'), 'w') as f:
                json.dump(self.config.__dict__, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config.debug:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            # with self.graph.as_default():
            #     self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            #     self.global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def infer(self):
        # This function is usually common to all your models
        pass

    def print_layer(self, layers, idx, name):
        if layers[-1][name].get_shape()[0].value is None:
            size = int(np.prod(layers[-1][name].get_shape()[1:]))
        else:
            size = int(np.prod(layers[-1][name].get_shape()))

        print("layer {} (high res) - {:>15}: {} [size {:,.0f}]" \
              "".format(len(layers), name, layers[idx][name].get_shape(), size / self.config.subbatch_max_size))

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def weight_variable(self, shape, layer_name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=self.config.initial_stddev, name='weight_{}'.format(layer_name))
        return tf.Variable(initial)

    def bias_variable(self, shape, layer_name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.zeros(shape, name='bias_{}'.format(layer_name))
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, dropout_rate, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            input_tensor = tf.nn.dropout(input_tensor, dropout_rate)
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim], layer_name)
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim], layer_name)
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            # activations = tf.nn.dropout(activations, droput_rate)
            return preactivate, activations

    def conv_layer(self, input_tensor, filter_size_3d, output_depth, stride, layer_name, act=tf.nn.relu, padded=True):
        if padded:
            input_tensor = self.circular_pad(input_tensor, filter_size_3d, use_r_padding=False)
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable(filter_size_3d + input_tensor.get_shape().as_list()[-1:] +
                                                [output_depth], layer_name)
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_depth], layer_name)
                self.variable_summaries(biases)
            with tf.name_scope('convolution'):
                preactivate = tf.nn.bias_add(tf.nn.conv3d(input_tensor, weights, stride, padding='VALID'), biases)
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def pool_layer(self, input_tensor, ksize, stride, layer_name, pool_type=tf.nn.avg_pool3d, padded=True):
        with tf.variable_scope(layer_name):
            if padded:
                input_tensor = tf_pad_wrap(input_tensor,
                                           [(0, 0), (0, 0), (0, 0), (ksize[3] // 2, ksize[3] // 2), (0, 0)])
            pool = pool_type(input_tensor, ksize=ksize, strides=stride, padding='VALID',
                             name='pool_{}'.format(layer_name))
            tf.summary.histogram('pooling', pool)
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


