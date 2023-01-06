# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
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
        self.metrics = pd.DataFrame()
        self.spike_times = sparse.csc_matrix(self.spikes.times)
        max_spikes = np.max(self.spike_times.sum(axis=1))
        # Create Unit objects.
        self._Units = []
        for uid in range(self.total_units):
            unit = Unit(uid, self.sync, self.spikes)
            self._Units.append(unit)
        
        # Define current index for iteration.
        self._current_unit_idx = 0

    def plot_aligned_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):
        """"""
        
        responses = self.get_aligned_response(self, stimulus, time_window, bin_size)
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        mean_response = np.nanmean(responses, axis=2)
        z_response = zscore(mean_response.T, axis=1)
        axs.imshow(z_response, cmap="binary")
        plt.show()

    def get_aligned_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):
        """"""
        #### Need to speed this up!
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        responses = np.zeros((num_bins, len(self), len(stimulus)))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.spike_times[:, window[0]:window[1]]
            for unit_idx, unit_resp in enumerate(resp):
                responses[:, unit_idx, event.index] = np.sum(unit_resp.reshape(num_bins, -1), axis=1).squeeze()
        return responses

    def plot_averaged_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):
        """"""

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        responses = np.zeros((num_bins, len(stimulus)))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.spike_times[:, window[0]:window[1]].sum(axis=0).squeeze()
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

    def sort(
            self, 
            metric_name, 
        ):
        """"""

        if metric_name in self.metrics.columns:
            sort_idx = np.argsort(self.metrics[metric_name])
            return sort_idx
            #### Need to reorder spike times matrix, _Units, metrics df
            # self._Units


    def add_metric(
            self, 
            metric_name, 
            metric_array, 
        ):
        """"""
        
        if len(metric_array) == len(self):
            self.metrics[metric_name] = metric_array
    
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
        if self._current_unit_idx < self.total_units:
            unit = self._Units[self._current_unit_idx]
            self._current_unit_idx += 1
            return unit

    def __len__(self):
        return len(self._Units)
        