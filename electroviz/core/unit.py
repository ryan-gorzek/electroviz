# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from electroviz.viz.psth import PSTH
from electroviz.viz.raster import Raster

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

    def plot_PSTH(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        PSTH(time_window, responses.mean(axis=0).squeeze())

    def plot_raster(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
            zscore=False, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        Raster(time_window, responses, ylabel="Stimulus Event", z_score=zscore)

    def get_response(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
        ):
        """"""

        sample_window = np.array(time_window) * 30
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size * 30))
        responses = np.zeros((len(stimulus), num_bins))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.get_spike_times(sample_window=window)
            bin_resp = resp.reshape((num_bins, -1)).sum(axis=1) / (bin_size / 1000)
            responses[event.index, :] = bin_resp
        return responses

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

    def get_spike_times(
            self, 
            sample_window=(None, None), 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Spikes.spike_times.tocsr()
            self.spike_times = spike_times_matrix[self.ID].tocsc()
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()
        