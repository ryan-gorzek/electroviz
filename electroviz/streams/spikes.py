
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
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
            kilosort_dict, 
        ):
        """"""

        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = int(np.max(kilosort_dict["spike_clusters"]) + 1)
        # Get the total number of samples in recording, passed from Imec data.
        self.total_samples = total_imec_samples
        # Build (sparse) spike times matrix in compressed sparse column format.
        self.spike_times = self._build_spike_times_matrix(kilosort_dict["spike_clusters"].squeeze(), 
                                                          kilosort_dict["spike_times"].squeeze())
        # Store Kilosort quality labels.
        self.cluster_quality = kilosort_dict["cluster_group"]
        # Get depth (along the probe) of each spike.
        self.cluster_depths = self._get_cluster_depths(kilosort_dict["templates"], kilosort_dict["whitening_mat_inv"], 
                                                       kilosort_dict["channel_positions"], kilosort_dict["spike_templates"], 
                                                       kilosort_dict["amplitudes"], kilosort_dict["spike_clusters"])


    def drop_and_rebuild(
            self, 
            drop_samples, 
        ):
        """"""

        # Create column (sample) logical index from sample indices.
        col_mask = np.ones(self.total_samples, dtype=bool)
        col_mask[drop_samples] = False
        self.total_samples += -len(drop_samples)
        self.spike_times = self.spike_times[:, col_mask]
        return None


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
        spike_times_matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=full_shape, dtype=bool)
        return spike_times_matrix


    def _get_cluster_depths(
            self, 
            templates, 
            inv_whitening_matrix, 
            channel_positions, 
            spike_templates, 
            amplitudes, 
            spike_clusters, 
        ):
        """"""

        # Unwhiten the spike templates.
        temps_unwhiten = np.empty(templates.shape)
        for idx, temp in enumerate(templates):
            temps_unwhiten[idx, :, :] = np.matmul(temp, inv_whitening_matrix)
        # 
        temp_channel_amps = temps_unwhiten.max(axis=1) - temps_unwhiten.min(axis=1)
        # 
        temp_amps_unsc = temp_channel_amps.max(axis=1)
        #
        thresh_vals = np.broadcast_to(temp_amps_unsc*0.3, tuple(reversed(temp_channel_amps.shape)))
        #
        temp_channel_amps[temp_channel_amps < thresh_vals.T] = 0
        #
        y_pos = channel_positions[:, 1]
        temp_depths = np.sum(temp_channel_amps*y_pos.T, axis=1)/np.sum(temp_channel_amps, axis=1)
        temp_chans = []
        for depth in temp_depths:
            dist = np.abs(y_pos - depth)
            (idx,) = np.where(dist == np.min(dist))
            try:
                temp_chans.append(idx[0])
            except:
                temp_chans.append(np.nan)
        spike_depths = np.array(temp_chans)[spike_templates]
        # Map spike depths to cluster depths.
        cluster_depths = []
        for cluster in range(self.total_units):
            if cluster in spike_clusters:
                cluster_idx = spike_clusters == cluster
                cluster_depths.append(np.nanmean(spike_depths[cluster_idx]))
            else:
                cluster_depths.append(np.nan)
        return cluster_depths

