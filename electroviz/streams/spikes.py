# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy import sparse

class Spikes:
    """

    """


    def __init__(
            self, 
            spike_clusters, 
            spike_times, 
            total_imec_samples, 
        ):
        """"""

        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = int(np.max(spike_clusters) + 1)
        # Get the total number of samples in recording, passed from Imec data.
        self.total_samples = total_imec_samples
        # Build (sparse) spike times matrix in compressed sparse row format.
        self.times = self._build_spike_times_matrix(spike_clusters, spike_times)

    def _build_spike_times_matrix(
            self, 
            spike_clusters, 
            spike_times, 
        ):
        """"""

        # Create a units-by-samples scipy sparse coordinate matrix to store spike times.
        full_shape = (self.total_units, self.total_samples)
        row_idx = spike_clusters
        col_idx = spike_times
        data = np.ones((spike_times.size,))
        spike_times_matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=full_shape)
        return spike_times_matrix

    # def _drop_spike_times(
    #         self, 
    #         drop_samples,
    #     ):
    #     """"""

    #     # Create column (sample) logical index from sample indices.
    #     col_mask = np.ones(self.total_samples, dtype=bool)
    #     col_mask[drop_samples] = False
    #     return self.spike_times[:, col_mask].tocoo()