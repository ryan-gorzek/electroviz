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
            total_imec_samples, 
            spike_clusters, 
            spike_times, 
            cluster_quality, 
        ):
        """"""

        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = int(np.max(spike_clusters) + 1)
        # Get the total number of samples in recording, passed from Imec data.
        self.total_samples = total_imec_samples
        # Build (sparse) spike times matrix in compressed sparse column format.
        self.times = self._build_spike_times_matrix(spike_clusters, spike_times)
        # Store Kilosort quality labels.
        self.cluster_quality = cluster_quality

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
        spike_times_matrix = sparse.csc_matrix((data, (row_idx, col_idx)), shape=full_shape, dtype=bool)
        return spike_times_matrix

    def drop_and_rebuild(
            self, 
            drop_samples, 
        ):
        """"""

        # Create column (sample) logical index from sample indices.
        col_mask = np.ones(self.total_samples, dtype=bool)
        col_mask[drop_samples] = False
        self.total_samples += -len(drop_samples)
        return self.times[:, col_mask]
