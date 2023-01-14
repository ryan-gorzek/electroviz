# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from scipy.stats import zscore

class Unit:
    """
    
    """
    
    def __init__(
            self, 
            unit_id, 
            imec_sync, 
            kilosort_spikes, 
        ):
        """"""
        
        self.ID = unit_id
        self._Sync = imec_sync
        self._Spikes = kilosort_spikes
        self.spike_times = np.empty((0, 0))
        self.kernels = []

    def add_kernel(
            self, 
            kernel, 
        ):
        """"""

        self.kernels.append(kernel)

    def plot_aligned_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        responses = np.zeros((num_bins, len(stimulus)))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.get_spike_times(sample_window=window)
            responses[:, event.index] = np.sum(resp.reshape(num_bins, -1), axis=1).squeeze()
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        z_response = zscore(responses.T, axis=1)
        axs.imshow(responses.T, cmap="binary")
        plt.show()

    def plot_averaged_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        responses = np.zeros((num_bins, len(stimulus)))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.get_spike_times(sample_window=window)
            responses[:, event.index] = np.sum(resp.reshape(num_bins, -1), axis=1).squeeze()
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        mean_response = np.nanmean(responses.squeeze(), axis=1)
        axs.axvline(50, color="k", linestyle="dashed")
        axs.plot(mean_response, color="b")
        axs.set_xlabel("Time from onset (ms)")
        axs.set_xticks([0, 50, 100, 150, 200, 250])
        axs.set_xticklabels([-50, 0, 50, 100, 150, 200])
        plt.show()

    def get_spike_times(
            self, 
            sample_window=[None, None], 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Spikes.spike_times.tocsr()
            self.spike_times = spike_times_matrix[self.ID].tocsc()
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()
        