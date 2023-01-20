
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
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
        self.sampling_rate = self._Sync.sampling_rate
        self.spike_times = np.empty((0, 0))


    def plot_PSTH(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            ax_in=None, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        PSTH(time_window, responses.mean(axis=0).squeeze(), ax_in=ax_in)


    def plot_raster(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            zscore=False, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        Raster(time_window, responses, ylabel="Stimulus Event", z_score=zscore)


    def plot_summary(
            self, 
            stimuli, 
            kernels, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(4, 5)

        self.plot_PSTH(stimuli[0], ax_in=axs[0][0])
        axs[0][0].set_xlabel("")
        axs[0][0].set_xticklabels([])
        kernels[0].plot_norm_delay(ax_in=axs[0][1])
        axs[0][1].set_xlabel("")
        axs[0][1].set_xticklabels([])
        kernels[0].plot_raw(ax_in=axs[0][2:5])

        self.plot_PSTH(stimuli[1], ax_in=axs[1][0])
        axs[1][0].set_xlabel("")
        axs[1][0].set_xticklabels([])
        kernels[1].plot_norm_delay(ax_in=axs[1][1])
        axs[1][1].set_xlabel("")
        axs[1][1].set_xticklabels([])
        kernels[1].plot_raw(ax_in=axs[1][2:5])

        self.plot_PSTH(stimuli[2], ax_in=axs[2][0])
        axs[2][0].set_xlabel("")
        axs[2][0].set_xticklabels([])
        kernels[2].plot_norm_delay(ax_in=axs[2][1])
        axs[2][1].set_xlabel("")
        axs[2][1].set_xticklabels([])
        axs[2][2].axis("off")
        kernels[2].plot_raw(ax_in=axs[2][3])
        axs[2][3].set_xlabel("")
        axs[2][3].set_xticklabels([])
        axs[2][4].axis("off")

        self.plot_PSTH(stimuli[3], ax_in=axs[3][0])
        kernels[3].plot_norm_delay(ax_in=axs[3][1])
        axs[3][2].axis("off")
        kernels[3].plot_raw(ax_in=axs[3][3])
        axs[3][4].axis("off")

        fig.suptitle("Unit #" + str(self.ID))
        plt.show(block=False)
        fig.set_size_inches(30, 15)


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
            metric_name, 
            metric, 
        ):
        """"""
        
        if not metric_name in self._Population.units.columns:
            self._Population.units[metric_name] = [np.nan]*self._Population.units.shape[0]
        (unit_idx,) = np.where(self._Population.units["unit_id"] == self.ID)[0]
        self._Population.units.at[unit_idx, metric_name] = metric


    def get_spike_times(
            self, 
            sample_window=(None, None), 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Spikes.spike_times.tocsr()
            self.spike_times = spike_times_matrix[self.ID].tocsc()
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()

