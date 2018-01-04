from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import random
from utils.batch_factory import BatchFactory
from utils.SparseGenerator import SparseGenerator
import glob
import argparse

import tensorflow as tf


# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
from models import make_model
from models import available_models
# from hpsearch import hyperband, randomsearch

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))


def main(_):
    # config = flags.FLAGS.__flags.copy()

    # fixed_params must be a string to be passed in the shell, let's use JSON
    # config["fixed_params"] = json.loads(config["fixed_params"])
    config.fixed_params = json.loads(config.fixed_params)

    high_res_protein_feature_filenames = sorted(
        glob.glob(os.path.join(config.data_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(
        glob.glob(os.path.join(config.data_dir, "*residue_features.npz")))

    validation_end = int(len(high_res_protein_feature_filenames) * (1. - config.test_fraction))
    train_end = validation_start = int(validation_end * (1. - config.validation_fraction))

    if config.debug:
        shuffle_opt = False
    else:
        shuffle_opt = True

    if not config.mode == 'infer' and not config.mode == 'test':
        train_data = SparseGenerator()
        train_data.load_data(high_res_protein_feature_filenames[:train_end],
                             high_res_grid_feature_filenames[:train_end])

        validation_data = SparseGenerator()
        validation_data.load_data(high_res_protein_feature_filenames[validation_start:validation_end],
                                  high_res_grid_feature_filenames[validation_start:validation_end])

    elif config.mode == 'test':
        test_data = SparseGenerator()
        test_data.load_data(high_res_protein_feature_filenames[validation_end:],
                            high_res_grid_feature_filenames[validation_end:])
    if config.fullsearch:
        pass
        # Some code for HP search ...
    elif config.dry_run:
        model = make_model(config)
    else:
        model = make_model(config)
        if config.mode == 'infer':
            model.infer()
        else:
            if config.mode == 'test':
                model.test(test_data)
            elif config.mode == 'train':
                model.train(train_data, validation_data)
                model.save('end')
            # No need to capture wrong mode, not allowed by argparser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''\t\t+-------------------------------------+
\t\t|                                     |
\t\t|              ProtNets               |
\t\t|                                     |
\t\t|         Neural networks for         |
\t\t|  protein spherical representations  |
\t\t|                                     |
\t\t|                                     |
\t\t+-------------------------------------+
''', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sparse',
                        action='store_true',
                        help='Turns on sparse representation for data input (default: %(default)s)')
    parser.add_argument('--fullsearch',
                        action='store_true',
                        help='Perform a full search of hyperparameter space ex:(hyperband > lr search > hyperband '
                             'with best lr) (default: %(default)s)')
    parser.add_argument('--dry_run',
                        action='store_true',
                        help='Perform a dry_run (default: %(default)s)')
    parser.add_argument('--nb_process',
                        default=4,
                        type=int,
                        help='Number of parallel process to perform a HP search (default: %(default)s)')

    # fixed_params is a trick I use to be able to fix some parameters inside the model random function
    # For example, one might want to explore different models fixing the learning rate, see the basic_model
    # get_random_config function
    parser.add_argument('--fixed_params',
                        default='{}',
                        type=str,
                        help='JSON inputs to fix some params in a HP search, ex: {"lr": 0.001} (default: %(default)s)')

    # Agent configuration
    parser.add_argument('--model_name',
                        default='CNN',
                        type=str,
                        choices=available_models,
                        help='Unique name of the model (default: %(default)s)')
    parser.add_argument('--optimizer',
                        default='Adam',
                        type=str,
                        choices=['Adam', 'Nesterov', 'AdaDelta'],
                        help='Model optimizer (default: %(default)s)')
    parser.add_argument('--best',
                        action='store_true',
                        help='Force to use the best known configuration (default: %(default)s)')
    parser.add_argument('--l2_beta',
                        default=0.001,
                        type=float,
                        help='Initial mean for the neural network (default: %(default)s)')
    parser.add_argument('--initial_stddev',
                        default=0.1,
                        type=float,
                        help='Initial standard deviation for the neural network (default: %(default)s)')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='The learning rate of SGD (default: %(default)s)')
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float,
                        help='Dropout value for training (default: %(default)s)')
    parser.add_argument('--no_batch_norm',
                        action='store_false',
                        help='Disable batch normalization (default: %(default)s)')

    # Environment configuration
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug mode (default: %(default)s)')
    parser.add_argument('--max_iter',
                        default=10,
                        type=int,
                        help='Number of training steps (default: %(default)s)')
    parser.add_argument('--mode',
                        type=str,
                        default='train',  # FIXME: Change back to infer when finished
                        choices=['train', 'test', 'infer'],
                        help='Infer from single data input (default: %(default)s)')

    # This is very important for TensorBoard
    # each model will end up in its own unique folder using time module
    # Obviously one can also choose to name the output folder
    parser.add_argument('--result_dir',
                        default=os.path.join(dir, 'results', str(int(time.time()))),
                        help='Name of the directory to store/log the model (if it exists, the model will be loaded '
                             'from it) (default: %(default)s)')
    parser.add_argument('--data_dir',
                        default=os.path.join(dir, 'data',  'culled_pc30', 'atomistic_features_spherical'),
                        type=str,
                        help='Name of the directory with the data for training and testing (default: %(default)s)')

    #  Another important point, you must provide an access to the random seed
    # to be able to fully reproduce an experiment
    parser.add_argument('--seed',
                        default=random.randint(0, sys.maxsize),
                        type=int,
                        help='Explicit value for dropout seed, otherwise random integer')

    # Data division
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
    parser.add_argument("--batch-size",
                        help="Maximum batch size used for gradient calculation (default: %(default)s)", type=int,
                        default=25)
    config = parser.parse_args()
    tf.app.run(main=main)
