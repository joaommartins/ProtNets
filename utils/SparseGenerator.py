import numpy as np


class SparseGenerator:
    def __init__(self):
        self.subbatch_size = 25
        self._current_batch = 0
        self._data = []
        self.grid_indexes = []
        self.charges = []
        self.masses = []
        self.aa_one_hot = []
        self._hold = True

        # self.shape = [self.subbatch_size, 24, 76, 151, 2]

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
                # self._data.append([[np.insert(grid_indexes, 3, 0, axis=1), np.insert(grid_indexes, 3, 1, axis=1)],
                #                    charges, masses, aa_one_hot])
        # self._data = np.array(self._data)
        # self.grid_indexes = np.array(self.grid_indexes)

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
            # shape = np.array([batch_size, 24, 76, 151, 2])
            shape = np.array([end-start, 24, 76, 151, 2])
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
            shape = np.array([end-start, 24, 76, 151, 2])
            # hots_index = np.array([[ind, np.argmax(x)] for ind, x in enumerate(self.aa_one_hot[start:end])])
            hots = np.array(self.aa_one_hot[start:end])
            return np.concatenate(indices), np.concatenate(values).ravel(), shape, hots

    def __repr__(self):
        return str(len(self.grid_indexes))

    def hold(self):
        return self._hold

    def restart(self):
        self._current_batch = 0
        self._hold = True
