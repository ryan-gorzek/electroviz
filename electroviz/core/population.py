# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import copy
import numpy as np
import pandas as pd
from scipy import sparse
from electroviz.core.unit import Unit
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from scipy.stats import zscore

class Population:
    """

    """

    def __init__(
            self, 
            imec, 
            kilosort, 
        ):
        """"""

        self._Sync = imec[0]
        self._Spikes = kilosort[0]
        self.total_samples = self._Spikes.total_samples
        self.total_units = self._Spikes.total_units
        self.spike_times = self._Spikes.spike_times
        # Create Unit objects.
        self._Units = []
        for uid in range(self.total_units):
            unit = Unit(uid, self._Sync, self._Spikes)
            self._Units.append(unit)
        # Populate unit metrics dataframe.
        self.units = pd.DataFrame()
        self.units["unit_id"] = np.arange(0, self.total_units)
        self.units["cluster_quality"] = self._Spikes.cluster_quality
        # self.units["depth"] = self._Spikes.spike_depths
        # firing rate, peak channel number
        self.depths = self._Spikes.cluster_depths
        # Define current index for iteration.
        self._current_Unit_idx = 0

    def plot_aligned_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
            cmap="binary", 
        ):
        """"""
        
        responses = self.get_aligned_response(stimulus, time_window, bin_size)
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        mean_response = np.nanmean(responses, axis=2)
        z_response = zscore(mean_response, axis=1)
        axs.imshow(z_response, cmap=cmap)
        axs.set_xlabel("Time from onset (ms)")
        axs.set_xticks([0, 50, 100, 150, 200, 250])
        axs.set_xticklabels([-50, 0, 50, 100, 150, 200])
        axs.set_ylabel("Unit")
        fig.set_size_inches(5, 11)
        plt.show()

    def get_aligned_response(
            self, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):
        """"""
        
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        responses = np.zeros((len(self), num_bins, len(stimulus)))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.spike_times[:, window[0]:window[1]].toarray()
            bin_resp = resp.reshape((len(self), num_bins, -1)).sum(axis=2)
            responses[:, :, event.index] = bin_resp
        return responses

    # def plot_averaged_response(
    #         self, 
    #         stimulus, 
    #         time_window=[-0.050, 0.200], 
    #         bin_size=0.001, 
    #     ):
    #     """"""

    #     sample_window = np.array(time_window)*30000
    #     num_samples = int(sample_window[1] - sample_window[0])
    #     num_bins = int(num_samples/(bin_size*30000))
    #     responses = np.zeros((num_bins, len(stimulus)))
    #     for event in stimulus:
    #         window = (sample_window + event.sample_onset).astype(int)
    #         resp = self.spike_times[:, window[0]:window[1]].sum(axis=0).squeeze()
    #         responses[:, event.index] = np.sum(resp.reshape(num_bins, -1), axis=1).squeeze()
    #     mpl_use("Qt5Agg")
    #     fig, axs = plt.subplots()
    #     mean_response = np.nanmean(responses.squeeze(), axis=1)
    #     axs.axvline(50, color="k", linestyle="dashed")
    #     axs.plot(mean_response, color="b")
    #     axs.set_xlabel("Time from onset (ms)")
    #     axs.set_xticks([0, 50, 100, 150, 200, 250])
    #     axs.set_xticklabels([-50, 0, 50, 100, 150, 200])
    #     plt.show()

    # def sort(
    #         self, 
    #         metric_name, 
    #     ):
    #     """"""

    #     if metric_name in self.metrics.columns:
    #         sort_idx = np.argsort(self.metrics[metric_name])
    #         return sort_idx
    #         #### Need to reorder spike times matrix, _Units, metrics df
    #         # self._Units


    # def add_metric(
    #         self, 
    #         metric_name, 
    #         metric_array, 
    #     ):
    #     """"""
        
    #     if len(metric_array) == len(self):
    #         self.metrics[metric_name] = metric_array
    
    def __getitem__(
            self, 
            input, 
        ):
        """"""

        if isinstance(input, int):
            subset = self._Units[unit_idx]
        else:
            try:
                subset = self._get_subset(input)
            except TypeError:
                print("Failed.")
        return subset

    def __iter__(self):
        return iter(self._Units)

    def __next__(self):
        if self._current_unit_idx < self.total_units:
            unit = self._Units[self._current_unit_idx]
            self._current_unit_idx += 1
            return unit

    def __len__(self):
        return len(self._Units)

    def _get_subset(
            self, 
            slice_or_array, 
        ):
        """"""

        subset = copy.copy(self)
        subset._Units = self._Units[slice_or_array]
        subset.spike_times = self.spike_times.tocsr()[slice_or_array, :].tocsc()
        subset.total_units = len(subset._Units)
        subset.units = self.units.iloc[slice_or_array]
        return subset
        