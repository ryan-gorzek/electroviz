
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import copy
import numpy as np
import pandas as pd
from scipy import sparse
from electroviz.core.unit import Unit
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from electroviz.viz.psth import PSTH
from electroviz.viz.raster import RateRaster
from scipy.stats import zscore
from math import remainder
from scipy.sparse import vstack

class Population:
    """

    """


    def __init__(
            self, 
            imec, 
            kilosort, 
        ):
        """"""

        self._Sync = imec
        self._Spikes = kilosort
        self.sampling_rate = self._Sync.sampling_rate
        self.total_samples = self._Spikes.total_samples
        self.total_units = self._Spikes.total_units
        self.spike_times = self._Spikes.spike_times.tocsc()
        # Create Unit objects.
        self._Units = []
        for unit_idx in range(self.total_units):
            unit_id = self._Spikes.cluster_id[unit_idx]
            peak_channel = self._Spikes.peak_channel[unit_idx]
            unit = Unit(unit_id, peak_channel, self._Sync, self._Spikes, self)
            self._Units.append(unit)
        # Populate unit metrics dataframe.
        self.units = pd.DataFrame()
        self.units["unit_id"] = self._Spikes.cluster_id
        self.units["quality"] = self._Spikes.cluster_quality
        self.units["peak_channel"] = self._Spikes.peak_channel.astype(int)
        self.units["depth"] = self._Spikes.cluster_depth.astype(float).astype(int)
        self.units["total_spikes"] = self.spike_times.getnnz(1)
        self.units["spike_rate"] = self.units["total_spikes"] / self._Sync.total_time
        # Define current index for iteration.
        self._current_Unit_idx = 0
        # Store data for raster plot.
        self._responses = None


    def __getitem__(
            self, 
            input, 
        ):
        """"""

        if isinstance(input, int):
            subset = self._Units[input]
        else:
            try:
                subset = self._get_subset(input)
            except TypeError:
                print("Failed.")
        return subset


    def __iter__(self):
        """"""

        return iter(self._Units)


    def __next__(self):
        """"""

        if self._current_unit_idx < self.total_units:
            unit = self._Units[self._current_unit_idx]
            self._current_unit_idx += 1
            return unit


    def __len__(self):
        """"""

        return len(self._Units)


    def __add__(self, other):
        """"""

        addpop = copy.deepcopy(self)
        addpop._Units = self._Units + other._Units
        for unit in addpop._Units:
            unit._Population = addpop
        # addpop.spike_times = vstack([self.spike_times.tocsr(), other.spike_times.tocsr()]).tocsc()
        addpop.total_units = len(self._Units) + len(other._Units)
        addpop.units = pd.concat((self.units, other.units), axis=0).reset_index(drop=True)
        return addpop

    def plot_PSTH(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            ax_in=None, 
            responses=None, 
        ):
        """"""

        if responses is None:
            self._responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        else:
            self._responses = responses
        PSTH(time_window, self._responses.mean(axis=0).squeeze(), ax_in=ax_in)


    def plot_raster(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
            fig_size=(6, 9), 
            save_path="", 
            ax_in=None, 
            responses=None, 
            return_ax=False, 
        ):
        """"""
        
        if responses is None:
            self._responses = self.get_response(stimulus, time_window, bin_size=bin_size)
            plot_responses = self._responses
        else:
            self._responses = responses
            plot_responses = responses
        
        if return_ax is True:
            fig, ax = RateRaster(time_window, plot_responses, ylabel="Unit", fig_size=fig_size, save_path=save_path, ax_in=ax_in, return_ax=True)
            return fig, ax
        else:
            RateRaster(time_window, plot_responses, ylabel="Unit", fig_size=fig_size, save_path=save_path, ax_in=ax_in)


    def plot_mean_waveforms(
            self, 
        ):
        """"""

        mean_waveforms = []
        for unit in self:
            waveforms = unit.get_waveforms()
            mean_waveforms.append(waveforms.mean(axis=0).squeeze())
        
        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        ax.plot(range(82), np.array(mean_waveforms).T)
        plt.show(block=False)


    def plot_lines(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
            save_path="", 
            ax_in=None, 
            line_color="k", 
            bound_color=(0.8, 0.8, 0.8, 0.8), 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        response_mean = responses.mean(axis=1).squeeze()
        response_sem = np.std(responses, axis=1) / np.sqrt(responses.shape[0])

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
        ax.plot(range(responses.shape[1]), response_mean, color=line_color)
        ax.fill_between(range(responses.shape[1]), response_mean, y2=response_mean + response_std, color=bound_color)
        ax.fill_between(range(responses.shape[1]), response_mean, y2=response_mean - response_std, color=bound_color)
        ax.set_xticks(np.linspace(0, responses.shape[1], 6))
        ax.set_xticklabels(np.linspace(*time_window, 6))
        ax.set_xlabel("Time from Stimulus Onset (ms)", fontsize=16)
        if ax_in is None:
            plt.show(block=False)


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
        responses = np.zeros((len(self), num_bins, len(stimulus)))
        for idx, event in enumerate(stimulus):
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.spike_times[:, window[0]:window[1]].toarray()
            bin_resp = resp.reshape((len(self), num_bins, -1)).sum(axis=2) / (bin_size / 1000)
            responses[:, :, idx] = bin_resp
        return responses.mean(axis=2)


    def get_mean_waveforms(
            self, 
            save_path="", 
            stat_in=None, 
        ):
        """"""


        def normalize(mean_waveform):
            trough = np.min(mean_waveform)
            peak = np.max(mean_waveform - trough)
            norm_waveform = (((mean_waveform - trough) / peak) * 2) - 1
            return norm_waveform


        for unit, stat in zip(self, stat_in):
            waveforms = unit.get_waveforms()
            if waveforms is not None:
                mean_waveform = waveforms.mean(axis=0).squeeze()
                trough, peak = np.min(mean_waveform), np.max(mean_waveform)
                pt_ratio = abs(peak) / abs(trough)
                trough_idx, peak_idx = np.where(mean_waveform == trough)[0][0], np.where(mean_waveform == peak)[0][0]
                pt_time = (peak_idx - trough_idx) / 30
                np.save(save_path + "/Unit" + str(unit.ID) + "_WaveformMean.npy", normalize(mean_waveform))
                np.save(save_path + "/Unit" + str(unit.ID) + "_WaveformStats.npy", np.array([pt_time, pt_ratio, stat]))


    def plot_corr_mat(
            self, 
        ):
        """"""
        
        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        corr_mat = np.corrcoef(self._bin_spikes())
        ax.imshow(corr_mat, cmap="RdBu_r", clim=[-1, 1])
        ax.set_xlabel("Unit")
        ax.set_ylabel("Unit")
        cax = inset_axes(ax, width="5%", height="90%", loc="center right", borderpad=-5)
        colorbar = fig.colorbar(ax.images[0], cax=cax)
        colorbar.set_label("Pearson Correlation Coefficient", rotation=-90)
        cax.yaxis.labelpad = 15
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cax.tick_params(labelsize=12)
        plt.subplots_adjust(left=0.12, right=0.75, top=0.95, bottom=0.1)
        plt.show(block=False)
        fig.set_size_inches(10, 8)


    def plot_rate_histogram(
            self, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        spike_rates = np.array(self.units["spike_rate"])
        plt.hist(spike_rates, bins=np.arange(0, spike_rates.max(), 5), density=True)
        plt.axvline(spike_rates.mean(), color="k", linestyle="--")
        axs.set_xlabel("Spikes/s", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)
        plt.show(block=False)
        fig.set_size_inches(8, 8)


    def sort(
            self, 
            metric, 
            order="descend", 
        ):
        """"""

        if metric in self.units.columns:
            sort_idx = np.argsort(self.remove(np.isnan(self.units[metric])).units[metric].to_numpy())
            if order == "ascend":
                subset = self.remove(np.isnan(self.units[metric]))._get_subset(sort_idx)
            elif order == "descend":
                subset = self.remove(np.isnan(self.units[metric]))._get_subset(sort_idx[::-1])
            return subset


    def remove(
            self, 
            idx, 
        ):
        """"""

        if isinstance(idx[0], np.bool_):
            (keep_idx,) = np.where(idx == False)
        else:
            keep_idx = idx
        subset = self._get_subset(np.array(keep_idx))
        return subset


    def _get_subset(
            self, 
            slice_or_array, 
        ):
        """"""

        subset = copy.deepcopy(self)
        subset._Units = list(np.array(self._Units)[slice_or_array])
        for unit in subset._Units:
            unit._Population = subset
        subset.spike_times = self.spike_times.tocsr()[slice_or_array, :].tocsc()
        subset.total_units = len(subset._Units)
        subset.units = self.units.iloc[slice_or_array].reset_index(drop=True)
        return subset


    def _bin_spikes(
            self, 
            bin_size=100, 
        ):
        """"""
        
        drop_end = int(self.total_samples % (bin_size * 30))
        num_bins = int((self.total_samples - drop_end)/(bin_size * 30))
        spike_times = self.spike_times[:, :-drop_end].toarray()
        spike_rate = spike_times.reshape((len(self), num_bins, -1)).sum(axis=2) / (bin_size / 1000)
        return spike_rate

