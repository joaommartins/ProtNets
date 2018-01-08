from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import BaseModel
import numpy as np
import tensorflow as tf
from utils.batch_factory import get_batch

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

class CNN(BaseModel):

    def _set_agent_props(self):
        # with self.graph.as_default():
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
                self.input = tf.placeholder(tf.float32,
                                            shape=[None] + self.input_shape,
                                            name='input_values')
                self.train_labels = tf.placeholder(tf.float32, shape=[None, self.output_size],
                                                    name='input_labels')
                self.dropout_keep_prob = tf.placeholder(tf.float32)
                layer = self.input
                self.layers.append({})
                self.layers[-1]['input'] = layer
                self.print_layer(self.layers, -1, 'input')

            # LAYER 1
            with tf.variable_scope('layer1'):
                self.layers.append({})
                # Convlayer1
                layer = self.conv_layer(layer, filter_size_3d=[4, 2, 5], output_depth=32,
                                        stride=[1, 1, 2, 2, 1], layer_name='convolution', auto_pad=False,
                                        explicit_pad=[0, 2, 2])
                self.layers[-1]['convolution'] = layer
                self.print_layer(self.layers, -1, 'convolution')
                # MaxPoolLayer1
                layer = self.pool_layer(layer, ksize=[1, 4, 2, 5, 1], stride=[1, 1, 2, 2, 1],
                                        layer_name='layer1', pool_type=tf.nn.avg_pool3d)
                self.layers[-1]['pooling'] = layer
                self.print_layer(self.layers, -1, 'pooling')

            # LAYER 2
            with tf.variable_scope('layer2'):
                self.layers.append({})
                # Convlayer2
                layer = self.conv_layer(layer, filter_size_3d=[2, 2, 4], output_depth=64,
                                        stride=[1, 1, 1, 1, 1], layer_name='convolution', auto_pad=False,
                                        explicit_pad=[0, 1, 1])
                self.layers[-1]['convolution'] = layer
                self.print_layer(self.layers, -1, 'convolution')
                # MaxPoolLayer2
                layer = self.pool_layer(layer, ksize=[1, 3, 3, 3, 1], stride=[1, 2, 2, 2, 1],
                                        layer_name='pooling', pool_type=tf.nn.avg_pool3d)
                self.layers[-1]['pooling'] = layer
                self.print_layer(self.layers, -1, 'pooling')

            # LAYER 3
            with tf.variable_scope('layer3'):
                self.layers.append({})
                # Convlayer3
                layer = self.conv_layer(layer, filter_size_3d=[2, 2, 3], output_depth=128,
                                        stride=[1, 1, 1, 2, 1], layer_name='convolution', auto_pad=False,
                                        explicit_pad=[0, 1, 1])
                self.layers[-1]['convolution'] = layer
                self.print_layer(self.layers, -1, 'convolution')
                # MaxPoolLayer3
                layer = self.pool_layer(layer, ksize=[1, 1, 3, 3, 1], stride=[1, 1, 1, 2, 1],
                                        layer_name='pooling', pool_type=tf.nn.avg_pool3d)
                self.layers[-1]['pooling'] = layer
                self.print_layer(self.layers, -1, 'pooling')

            # LAYER 4
            with tf.variable_scope('layer4'):
                self.layers.append({})
                # Convlayer4
                layer = self.conv_layer(layer, filter_size_3d=[3, 7, 3], output_depth=256,
                                        stride=[1, 1, 2, 1, 1], layer_name='convolution', auto_pad=False,
                                        explicit_pad=[0, 2, 1])
                self.layers[-1]['convolution'] = layer
                self.print_layer(self.layers, -1, 'convolution')
                # MaxPoolLayer4
                layer = self.pool_layer(layer, ksize=[1, 1, 3, 3, 1], stride=[1, 1, 1, 1, 1],
                                        layer_name='pooling', pool_type=tf.nn.avg_pool3d)
                self.layers[-1]['pooling'] = layer
                self.print_layer(self.layers, -1, 'pooling')

            # FC1
            self.layers.append({})
            _, layer, weight = self.nn_layer(layer, 2048, 'FC1', dropout_keep_prob=self.dropout_keep_prob, conv2fc=True)
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['FC1'] = layer
            self.print_layer(self.layers, -1, 'FC1')

            # FC2
            self.layers.append({})
            _, layer, weight = self.nn_layer(layer, 2048, 'FC2',
                                             dropout_keep_prob=self.dropout_keep_prob)
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['FC2'] = layer
            self.print_layer(self.layers, -1, 'FC2')

            # OUTPUT LAYER (NAME MUST BE "output_layer" for loss to work)
            self.layers.append({})
            preactivated, output_layer, weight = self.nn_layer(layer, self.output_size, 'output_layer',
                                                               dropout_keep_prob=1.0, act=tf.nn.softmax)
            self.layers[-1]['preactivation'] = preactivated
            self.layers[-1]['W'] = weight
            self.print_layer(self.layers, -1, 'W')
            self.layers[-1]['output_layer'] = output_layer
            self.print_layer(self.layers, -1, 'output_layer')
            # return graph

    def infer(self):
        pass
