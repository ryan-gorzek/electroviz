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
            population, 
        ):
        """"""
        
        self.ID = unit_id
        self._Sync = imec_sync
        self._Spikes = kilosort_spikes
        self._Population = population
        self.spike_times = np.empty((0, 0))
        self.kernels = []

    def add_metric(
            self, 
            unit_id, 
            metric_name, 
            metric, 
        ):
        """"""
        
        if not metric_name in self.units.columns:
            self._Population.units[metric_name] = [np.nan]*self._Population.units.shape[0]
        (unit_idx,) = np.where(self._Population.units["unit_id"] == self.ID)
        self.units.at[unit_idx[0], metric_name] = metric

    def add_kernel(
            self, 
            kernel, 
        ):
        """"""

        self.kernels.append(kernel)

    def plot_raster(
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

    def plot_PSTH(
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

    def get_spike_times(
            self, 
            sample_window=(None, None), 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Spikes.spike_times.tocsr()
            self.spike_times = spike_times_matrix[self.ID].tocsc()
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()
        