import os
import glob
import argparse
import numpy as np

from utils.batch_factory import BatchFactory

import tensorflow as tf



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if __name__ == '__main__':
    r, theta, phi, channels = [24, 76, 151, 2]

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

    train_data = BatchFactory()
    train_data.add_data_set("high_res",
                            high_res_protein_feature_filenames[:train_end],
                            high_res_grid_feature_filenames[:train_end],
                            duplicate_origin=config.duplicate_origin)

    train_data.add_data_set("model_output",
                            high_res_protein_feature_filenames[:train_end],
                            key_filter=[config.data_type + "_one_hot"])

    # validation_data = BatchFactory()
    # validation_data.add_data_set("high_res",
    #                              high_res_protein_feature_filenames[validation_start:validation_end],
    #                              high_res_grid_feature_filenames[validation_start:validation_end],
    #                              duplicate_origin=config.duplicate_origin)
    # validation_data.add_data_set("model_output",
    #                              high_res_protein_feature_filenames[validation_start:validation_end],
    #                              key_filter=[config.data_type + "_one_hot"])
    # test_data = BatchFactory()
    # test_data.add_data_set("high_res",
    #                        high_res_protein_feature_filenames[validation_end:],
    #                        high_res_grid_feature_filenames[validation_end:],
    #                        duplicate_origin=config.duplicate_origin)
    # test_data.add_data_set("model_output",
    #                        high_res_protein_feature_filenames[validation_end:],
    #                        key_filter=[config.data_type + "_one_hot"])


    tfrecords_filename = os.path.join(config.data_dir, 'training_set.tfrecord')

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    letf_data = train_data.data_size()

    more_data = True
    while more_data:
        batch, gradient_batch_sizes = train_data.next(1000,
                                                      subbatch_max_size=1000,
                                                      enforce_protein_boundaries=False)
        more_data = (train_data.feature_index != 0)
        x = batch['high_res'].tostring()
        y = batch['model_output'].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'examples': _int64_feature(1000),
            'r': _int64_feature(r),
            'theta': _int64_feature(theta),
            'phi': _int64_feature(phi),
            'image_raw': _bytes_feature(x),
            'mask_raw': _bytes_feature(y)}))
        writer.write(example.SerializeToString())
        letf_data -= 1000
        # if letf_data % 1000 == 0:
        print 'Data left: {}'.format(letf_data)
    writer.close()
