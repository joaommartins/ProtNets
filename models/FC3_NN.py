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
    def _set_agent_props(self):
        if self.config.data_type == 'aa':
            self.output_size = 21
        else:
            self.output_size = 3
        self.input_shape = [24, 76, 151, 2]

    def _get_best_config(self):
        return {}

    @staticmethod
    def _get_random_config(fixed_params={}):
        pass

    def _build_graph(self):
        with self.graph.as_default():
            # LAYER 0
            with tf.variable_scope('inputs'):
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')

                self.indices = tf.placeholder(tf.int32)
                self.values = tf.placeholder(tf.float32)
                self.shape = [self.config.batch_size] + self.input_shape
                self.input = tf.sparse_to_dense(self.indices, self.shape, self.values, validate_indices=False)

                self.train_labels = tf.placeholder(tf.float32, shape=[None, self.output_size],
                                                   name='input_labels')
                self.dropout_keep_prob = tf.placeholder(tf.float32)
                layer = self.input
                self.layers.append({})
                self.layers[-1]['input'] = layer
                self.print_layer(self.layers, -1, 'input')

                # FC1
            self.layers.append({})
            _, layer, weight = self.nn_layer(layer, 2048, 'FC1', dropout_keep_prob=self.dropout_keep_prob,
                                             conv2fc=True
                                             , batch_normalize=self.batch_normalization)
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['FC1'] = layer
            self.print_layer(self.layers, -1, 'FC1')

            # FC2
            self.layers.append({})
            _, layer, weight = self.nn_layer(layer, 2048, 'FC2', dropout_keep_prob=self.dropout_keep_prob,
                                             batch_normalize=self.batch_normalization)
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['FC2'] = layer
            self.print_layer(self.layers, -1, 'FC2')

            # OUTPUT LAYER (NAME MUST BE "output_layer" for loss to work)
            self.layers.append({})
            preactivated, output_layer, weight = self.nn_layer(layer, self.output_size, 'output_layer',
                                                               dropout_keep_prob=1.0, act=tf.nn.softmax,
                                                               batch_normalize=self.batch_norm)
            self.layers[-1]['preactivation'] = preactivated
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['output_layer'] = output_layer
            self.print_layer(self.layers, -1, 'output_layer')

    def infer(self):
        pass
