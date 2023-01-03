# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

class Unit:
    """
    docstring
    """
    
    def __init__(
            self, 
            unit_id, 
            imec_sync, 
            kilosort_spikes, 
        ):
        """"""
        
        self.unit_id = unit_id
        self.sync = imec_sync
        self.spikes = kilosort_spikes
        self.spike_times = np.empty((0, 0))

    def get_spike_times(
            self, 
            sample_window=[None, None], 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self.spikes.times.tocsr()
            self.spike_times = spike_times_matrix[self.unit_id].tocsc()
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()

        