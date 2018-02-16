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
    aa1 = "ACDEFGHIKLMNPQRSTVWYX"
    data_dir = '/Users/jmartins/PycharmProjects/ProtNets/data/infer'

    # fixed_params must be a string to be passed in the shell, let's use JSON
    high_res_protein_feature_filenames = sorted(
        glob.glob(os.path.join(data_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(
        glob.glob(os.path.join(data_dir, "*residue_features.npz")))

    infer_data = SparseGenerator()
    infer_data.load_data(high_res_protein_feature_filenames,
                         high_res_grid_feature_filenames)

    diff = []
    for index in range(1, 805):
        start = timer()
        indices, values, hots = infer_data.infer(index)
        print index, aa1[np.argmax(hots)]
        # print indices
        # print values.shape
        # print hots.shape
        end = timer()
        diff.append(end-start)

    print 'Average loop: {}'.format(np.average(diff))
