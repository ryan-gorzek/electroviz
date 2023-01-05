# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy import sparse
from electroviz.core.unit import Unit

class Population:
    """

    """

    def __init__(
            self, 
            imec, 
            kilosort, 
        ):
        """"""

        self.sync = imec[0]
        self.spikes = kilosort[0]
        self.total_samples = self.spikes.total_samples
        self.total_units = self.spikes.total_units
        self.spike_times = sparse.csc_matrix(self.spikes.times)
        max_spikes = np.max(self.spike_times.sum(axis=1))
        # Create Unit objects.
        self._Units = []
        for uid in range(self.total_units):
            unit = Unit(uid, self.sync, self.spikes)
            self._Units.append(unit)
        
        # Define current index for iteration.
        self._current_unit_idx = 0
    
    def __getitem__(
            self, 
            unit_idx, 
        ):
        """"""

        unit = self._Units[unit_idx]
        return unit

    def __iter__(self):
        return iter(self._Units)

    def __next__(self):
        """"""
        if self._current_unit_idx < self.total_units:
            unit = self._Units[self._current_unit_idx]
            self._current_unit_idx += 1
            return unit

    def __len__(self):
        return len(self._Units)
        