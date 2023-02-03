
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy import sparse
from scipy.stats import mode

class Spikes:
    """

    """


    def __init__(
            self, 
            total_imec_samples, 
            kilosort_dict, 
        ):
        """"""

        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = np.unique(kilosort_dict["spike_clusters"]).size
        # Store the cluster IDs.
        self.cluster_id = kilosort_dict["cluster_id"]
        # Store Kilosort quality labels.
        self.cluster_quality = kilosort_dict["cluster_quality"]
        # Get the total number of samples in recording, passed from Imec data.
        self.total_samples = total_imec_samples
        # Build (sparse) spike times matrix in compressed sparse column format.
        self.spike_times = self._build_spike_times_matrix(kilosort_dict["spike_clusters"].squeeze(), 
                                                          kilosort_dict["spike_times"].squeeze())
        self.peak_channel = kilosort_dict["peak_channel"].astype(int)
        self.cluster_depth = kilosort_dict["depth"].astype(float).astype(int)
        self.phy_path = kilosort_dict["phy_path"]
        self._remove_noise()


    def _build_spike_times_matrix(
            self, 
            spike_clusters, 
            spike_times, 
        ):
        """"""

        # Create a units-by-samples scipy sparse coordinate matrix to store spike times.
        full_shape = (spike_clusters.max() + 1, self.total_samples)
        row_idx = spike_clusters
        col_idx = spike_times
        data = np.ones((spike_times.size,))
        spike_times_matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=full_shape, dtype=bool)
        mask = np.array([clust for clust in np.arange(0, spike_clusters.max() + 1, 1) if clust in spike_clusters])
        spike_times_matrix = spike_times_matrix[mask, :]
        return spike_times_matrix


    def _remove_noise(
            self, 
        ):
        """"""

        noise_mask = self.cluster_quality != "noise"
        self.total_units = int(np.sum(noise_mask))
        self.cluster_id = self.cluster_id[noise_mask]
        self.cluster_quality = self.cluster_quality[noise_mask]
        self.spike_times = self.spike_times[noise_mask, :]
        self.peak_channel = self.peak_channel[noise_mask]
        self.cluster_depth = self.cluster_depth[noise_mask]

