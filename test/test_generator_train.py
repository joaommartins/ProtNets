# -*- coding: utf-8 -*-
# Copyright [2018] [João Martins]
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

import sys
sys.path.append('../')
from utils.SparseGenerator import SparseGenerator
import glob
import os
from timeit import default_timer as timer
import numpy as np

def test_load_sample_data():
    data_dir = '/Users/jmartins/PycharmProjects/ProtNets/data/culled_pc30/atomistic_features_spherical/'
    batch_size = 10
    validation_fraction = 0.10
    test_fraction = 0.10

    # fixed_params must be a string to be passed in the shell, let's use JSON
    high_res_protein_feature_filenames = sorted(
        glob.glob(os.path.join(data_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(
        glob.glob(os.path.join(data_dir, "*residue_features.npz")))

    train_data = SparseGenerator()
    train_data.load_data(high_res_protein_feature_filenames,
                         high_res_grid_feature_filenames)

    diff = []
    while train_data.hold():
        start = timer()
        indices, values, shape, hots_index = train_data.next(batch_size)
        end = timer()
        diff.append(end-start)

    print 'Average loop: {}'.format(np.average(diff))
