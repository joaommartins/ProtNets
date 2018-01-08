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

from models import BaseModel
import numpy as np
import tensorflow as tf
from utils.batch_factory import get_batch


class FC3_NN(BaseModel):

    def set_agent_props(self):
        self.layers = []
        if self.config.data_type == 'aa':
            self.output_size = 21
        else:
            self.output_size = 3
        self.input_shape = [24, 76, 151, 2]

    def get_best_config(self):
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        pass

    def build_graph(self):
        with self.graph.as_default():

            # LAYER 0
            input_scope = tf.VariableScope(reuse=False, name="inputs")
            with tf.variable_scope(input_scope):
                self.input = tf.placeholder(tf.float32,
                                            shape=[self.config.subbatch_max_size] + self.input_shape,
                                            name='input_values')
                self.train_labels = tf.placeholder(tf.int64, shape=[self.config.subbatch_max_size, self.output_size],
                                                    name='input_labels')
                self.input_layer = tf.reshape(self.input, (self.config.subbatch_max_size, np.prod(self.input_shape)))
                self.layers.append(self.input_layer)
                self.print_layer(self.layers, -1, 'Input layer')

            # LAYER 0
            self.layer1 = self.nn_layer(self.input_layer, self.input_layer.get_shape().as_list()[1], 128, 'layer1')
            self.layers.append(self.layer1)
            self.print_layer(self.layers, -1, 'layer1 RELU')

            # LAYER 1
            self.layer2 = self.nn_layer(self.layer1, self.layer1.get_shape().as_list()[1], 64, 'layer2')
            self.layers.append(self.layer2)
            self.print_layer(self.layers, -1, 'layer2 RELU')

            # OUTPUT LAYER
            self.output_layer = self.nn_layer(self.layer2, self.layer2.get_shape().as_list()[1], self.output_size,
                                        'output_layer', act=tf.nn.softmax)
            self.layers.append(self.output_layer)
            self.print_layer(self.layers, -1, 'Output Layer')

    def infer(self):
        pass

    # def learn_from_epoch(self, grid_matrix, labels, gradient_batch_sizes):
    #     for sub_iteration, (index, length) in enumerate(
    #             zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes)):
    #         grid_matrix_batch, labels_batch = get_batch(index, index + length, grid_matrix, labels)
    #
    #
    #         # feed_dict = dict({self.x_high_res: grid_matrix_batch,
    #         #                   self.y: labels_batch,
    #         #                   self.dropout_keep_prob: dropout_keep_prob})
    #         feed_dict = dict({self.input: grid_matrix_batch,
    #                           self.train_labels: labels_batch})
    #
    #         # print self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
    #         accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
    #         # accuracy = self.error_rate(eval_result, labels_batch)
    #         _, loss_value, summary, self.global_step = self.sess.run([self.train_step, self.loss, self.merged_summaries,
    #                                                                   self.global_step_var], feed_dict=feed_dict)
    #
    #         self.sw.add_summary(summary, global_step=self.global_step)
    #
    #         print('Loss step {}: {:.2f} [Subbatch error: {:.0f}%]'.format(self.global_step, loss_value,
    #                                                                       100-(100*accuracy)))
    #
    # def eval_from_epoch(self, grid_matrix, labels, gradient_batch_sizes):
    #     losses = []
    #     for sub_iteration, (index, length) in enumerate(
    #             zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes)):
    #         grid_matrix_batch, labels_batch = get_batch(index, index + length, grid_matrix, labels)
    #
    #         feed_dict = dict({self.input: grid_matrix_batch,
    #                           self.train_labels: labels_batch})
    #
    #         # eval_result = self.sess.run([self.layers[-1]], feed_dict=feed_dict)
    #         # error = self.error_rate(eval_result, labels_batch)
    #         accuracy = self.sess.run(self.test_accuracy, feed_dict=feed_dict)
    #         losses.append(accuracy)
    #         # print('Validation subbatch error: {:.0f}%'.format(100-(100*accuracy)))
    #     return losses
