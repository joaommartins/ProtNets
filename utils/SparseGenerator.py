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

import numpy as np

class SparseGenerator:
    def __init__(self):
        self._current_batch = 0
        self._data = []
        self.grid_indexes = []
        self.charges = []
        self.masses = []
        self.aa_one_hot = []
        self._hold = True

    def load_data(self, protein_feature, grid_feature):
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

    def next(self, batch_size):
        self._current_batch += 1
        if self._current_batch * batch_size >= len(self.grid_indexes):
            self._hold = False
            start, end = ((self._current_batch - 1) * batch_size, len(self.grid_indexes)) # FIXME: Change to wrap around
            start = -batch_size
            indices = [np.insert(x, 0, ind, axis=1) for ind, x in enumerate(self.grid_indexes[start:end])]
            indices = np.array([np.concatenate((np.insert(x, 4, 0, axis=1),
                                                np.insert(x, 4, 1, axis=1))) for ind, x in enumerate(indices)])
            values = np.array([np.concatenate((self.masses[start:end][index], self.charges[start:end][index]))
                               for index, val in enumerate(self.masses[start:end])])
            shape = np.array([end-start, 24, 76, 151, 2])
            hots = np.array(self.aa_one_hot[start:end])
            return np.concatenate(indices), np.concatenate(values).ravel(), shape, hots
        else:
            start, end = ((self._current_batch-1) * batch_size, self._current_batch * batch_size)
            indices = [np.insert(x, 0, ind, axis=1) for ind, x in enumerate(self.grid_indexes[start:end])]
            indices = np.array([np.concatenate((np.insert(x, 4, 0, axis=1),
                                np.insert(x, 4, 1, axis=1))) for ind, x in enumerate(indices)])
            values = np.array([np.concatenate((self.masses[start:end][index], self.charges[start:end][index]))
                               for index, val in enumerate(self.masses[start:end])])
            shape = np.array([end-start, 24, 76, 151, 2])
            hots = np.array(self.aa_one_hot[start:end])
            return np.concatenate(indices), np.concatenate(values).ravel(), shape, hots

    def __repr__(self):
        return str(len(self.grid_indexes))

    def hold(self):
        return self._hold

    def restart(self):
        self._current_batch = 0
        self._hold = True

    def grid_shape(self):
        return [24, 76, 151, 2]  # FIXME: ADD CALL TO CLASS INSTANTIATION

    def infer(self, residue_index):
        zeroth_index = residue_index - 1
        # zeroth_index = residue_index
        indices = np.insert(self.grid_indexes[zeroth_index], 0, 0, axis=1)
        # indices = np.insert(indices, 4, self.masses[zeroth_index], axis=1)
        indices = np.array([(np.insert(x, 4, 0), np.insert(x, 4, 1)) for ind, x in enumerate(indices)])
        values = np.array([np.concatenate((self.masses[zeroth_index][index], self.charges[zeroth_index][index]))
                           for index, val in enumerate(self.masses[zeroth_index])])
        # values = np.array([len(self.masses[zeroth_index]) * [1]])
        hots = np.array(self.aa_one_hot[zeroth_index])
        return np.concatenate(indices), np.concatenate(values).ravel(), hots
