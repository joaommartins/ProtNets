# from scipy.sparse import load_npz
import argparse
import os
import glob
import numpy as np
import tensorflow as tf


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class SparseGenerator:
    def __init__(self):
        self.subbatch_size = 10
        self._current_batch = 0
        self._hold = True
        self.shape = [self.subbatch_size, 24, 76, 151, 2]

    def load_data(self, protein_feature, grid_feature):
        self._data = []
        self.grid_indexes = []
        self.charges = []
        self.masses = []
        self.aa_one_hot = []
        for index, prot in enumerate(protein_feature):
            loader = np.load(prot)
            data = dict(zip((k for k in loader), (loader[k] for k in loader)))
            grid_loader = np.load(grid_feature[index])
            grid_data = dict(zip((k for k in grid_loader), (grid_loader[k] for k in grid_loader)))
            for dex, value in enumerate(grid_data['indices']):
                aa_one_hot = data['aa_one_hot'][dex]
                indexer = grid_data['selector'][dex][grid_data['selector'][dex] >= 0]
                grid_indexes = grid_data['indices'][dex][:len(indexer)]
                charges = data['charges'][indexer]
                masses = data['masses'][indexer]
                self.grid_indexes.append(grid_indexes)
                self.charges.append(charges)
                self.masses.append(masses)
                self.aa_one_hot.append(aa_one_hot)
                # self._data.append([[np.insert(grid_indexes, 3, 0, axis=1), np.insert(grid_indexes, 3, 1, axis=1)],
                #                    charges, masses, aa_one_hot])
        # self._data = np.array(self._data)
        # self.grid_indexes = np.array(self.grid_indexes)

    def next(self, batch_size):
        self._current_batch += 1
        if self._current_batch * batch_size >= len(self.grid_indexes):
            start, end = ((self._current_batch - 1) * batch_size, len(self.grid_indexes))
            self._hold = False
            indices = [np.insert(x, 0, ind, axis=1) for ind, x in enumerate(self.grid_indexes[start:end])]
            indices = np.array([np.concatenate((np.insert(x, 4, 0, axis=1),
                                                np.insert(x, 4, 1, axis=1))) for ind, x in enumerate(indices)])
            values = np.array([np.concatenate((self.masses[start:end][index], self.charges[start:end][index]))
                               for index, val in enumerate(self.masses[start:end])])
            shape = self.shape
            # hots_index = np.array([[ind, np.argmax(x)] for ind, x in enumerate(self.aa_one_hot[start:end])])
            hots = np.array(self.aa_one_hot[start:end])
            return np.concatenate(indices), np.concatenate(values).ravel(), shape, hots
        else:
            start, end = ((self._current_batch-1) * batch_size, self._current_batch * batch_size)
            indices = [np.insert(x, 0, ind, axis=1) for ind, x in enumerate(self.grid_indexes[start:end])]
            indices = np.array([np.concatenate((np.insert(x, 4, 0, axis=1),
                                np.insert(x, 4, 1, axis=1))) for ind, x in enumerate(indices)])
            values = np.array([np.concatenate((self.masses[start:end][index], self.charges[start:end][index]))
                               for index, val in enumerate(self.masses[start:end])])
            shape = self.shape
            # hots_index = np.array([[ind, np.argmax(x)] for ind, x in enumerate(self.aa_one_hot[start:end])])
            hots = np.array(self.aa_one_hot[start:end])
            return np.concatenate(indices), np.concatenate(values).ravel(), shape, hots

    def __repr__(self):
        return str(len(self.grid_indexes))

    def hold(self):
        return self._hold


def nn_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu, conv2fc=False):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # input_dim = np.prod(input_tensor.dense_shape[1:])
    # print input_dim
    # input_dim = np.prod(input_tensor.get_shape().as_list()[1:])
    input_dim = np.prod([24, 76, 151, 2])
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(layer_name):
        if conv2fc:
            # input_tensor = tf.sparse_reshape(
            #     input_tensor, (-1, np.prod([24, 76, 151, 2])))
            input_tensor = tf.reshape(
                    input_tensor, (-1, input_dim))
        # This Variable will hold the state of the weights for the layer
        weights = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
        biases = tf.truncated_normal([output_dim], stddev=0.1)
        preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        activations = act(preactivate, name='activation')
        return preactivate, activations, weights


def main(_):
    generator = SparseGenerator()
    generator.load_data(high_res_protein_feature_filenames[:train_end], high_res_grid_feature_filenames[:train_end])

    x = tf.sparse_placeholder(tf.float32)
    y = tf.placeholder(tf.int8)
    new_x = tf.sparse_tensor_to_dense(x)
    print new_x.get_shape()
    _, layer1, _ = nn_layer(new_x, 128, 'layer1', conv2fc=True)
    for_loss, output_layer, _ = nn_layer(layer1, 21, 'output')
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=for_loss, labels=y)
    loss = tf.reduce_mean(entropy)
    train_step = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)
    with tf.Session() as sess:

        # batch = generator.next(10)
        while generator.hold():
            indices, values, shape, hots_index = generator.next(10)
            print hots_index.shape
            print sess.run(train_step, feed_dict={
                x: (indices, values, shape),
                y: hots_index})


if __name__ == '__main__':
    # shape = [24, 76, 151, 2]

    curr_directory = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=os.path.join(curr_directory, 'data', 'culled_pc30', 'atomistic_features_spherical'),
                        type=str,
                        help='Name of the directory with the data for training and testing (default: %(default)s)')
    parser.add_argument('--validation_fraction',
                        default=0.10,
                        type=float,
                        help='Validation data fraction from train set (default: %(default)s)')
    parser.add_argument('--test_fraction',
                        default=0.10,
                        type=float,
                        help='Validation data fraction from train set (default: %(default)s)')
    parser.add_argument('--data_type',
                        default='aa',
                        type=str,
                        help='Data type to be trained on (default: %(default)s)',
                        choices=['aa', 'ss'])
    parser.add_argument("--duplicate_origin",
                        action='store_true',
                        help="Whether to duplicate the atoms in all bins at the origin for the spherical model")
    parser.add_argument("--max-batch-size",
                        help="Maximum batch size used during training (default: %(default)s)", type=int, default=1000)
    parser.add_argument("--subbatch-max-size",
                        help="Maximum batch size used for gradient calculation (default: %(default)s)", type=int,
                        default=25)

    config = parser.parse_args()

    high_res_protein_feature_filenames = sorted(
        glob.glob(os.path.join(config.data_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(
        glob.glob(os.path.join(config.data_dir, "*residue_features.npz")))

    validation_end = int(len(high_res_protein_feature_filenames) * (1. - config.test_fraction))
    train_end = validation_start = int(validation_end * (1. - config.validation_fraction))

    tf.app.run(main=main)
