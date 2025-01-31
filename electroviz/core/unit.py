
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from electroviz.viz.psth import PSTH
from electroviz.viz.raster import SpikeRaster
from electroviz.viz.summary import UnitSummary
from phylib.io.model import load_model
from scipy.stats import zscore

class Unit:
    """
    
    """


    def __init__(
            self, 
            unit_id, 
            peak_channel, 
            imec_sync, 
            kilosort_spikes, 
            population, 
        ):
        """"""
        
        self.ID = unit_id
        self.peak_channel = peak_channel
        self._Sync = imec_sync
        self._Spikes = kilosort_spikes
        self._Population = population
        self.total_samples = self._Population.total_samples
        self.sampling_rate = self._Sync.sampling_rate
        self._phy_path = self._Spikes.phy_path
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
            ax_in=None, 
        ):
        """"""

        spikes = self.get_spikes(stimulus, time_window)
        SpikeRaster(time_window, spikes, ylabel="Stimulus Event", ax_in=ax_in)


    def plot_summary(
            self, 
            stimuli, 
            kernels, 
        ):
        """"""

        UnitSummary(self, stimuli, kernels)


    def plot_waveforms(
            self, 
            ax_in=None, 
        ):
        """"""

        waveforms = self.get_waveforms()
        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
        if waveforms is not None:
            ax.plot(range(82), waveforms[::50, :].T, color=(0.7, 0.7, 0.7, 0.05), linewidth=2)
            mean_waveform = waveforms.T.mean(axis=1)
            ax.plot(range(82), mean_waveform, color=(0.4, 0.4, 0.4, 0.7), linewidth=4)
            ax.set_xlim((0, 81))
            ax.set_xticks(np.linspace(10, 71, 5))
            ax.set_xticklabels(np.linspace(-1, 1, 5))
            ax.set_xlabel("Time from Spike (ms)")
            peak, trough = np.max(mean_waveform), np.min(mean_waveform)
            wv_range = peak - trough
            ax.set_ylim((trough - 1.25*wv_range, peak + 1.25*wv_range))
        else:
            ax.axis("off")
        if ax_in is None:
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
            nan_idx = [], 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        response_mean = responses.mean(axis=0).squeeze()
        response_mean -= response_mean[:5].mean()
        response_sem = np.std(responses, axis=0) / np.sqrt(responses.shape[0])
        if len(nan_idx) > 0:
            response_mean[nan_idx.astype(int)] = np.nan
            response_sem[nan_idx.astype(int)] = np.nan

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
        ax.plot(range(responses.shape[1]), response_mean, color=line_color)
        ax.fill_between(range(responses.shape[1]), response_mean, y2=response_mean + response_sem, color=bound_color)
        ax.fill_between(range(responses.shape[1]), response_mean, y2=response_mean - response_sem, color=bound_color)
        ax.set_xticks(np.linspace(0, responses.shape[1], 6))
        ax.set_xticklabels(np.linspace(*time_window, 6))
        ax.set_xlabel("Time from Onset (ms)", fontsize=16)
        if ax_in is None:
            plt.show(block=False)


    def get_response(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
        ):
        """"""

        time_window = (time_window[0], time_window[1] + bin_size)
        sample_window = np.array(time_window) * 30
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size * 30))
        responses = np.zeros((len(stimulus), num_bins))
        for idx, event in enumerate(stimulus):
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.get_spike_times(sample_window=window)
            bin_resp = resp.reshape((num_bins, -1)).sum(axis=1) / (bin_size / 1000)
            responses[idx, :] = bin_resp
        return responses


    def get_spikes(
            self, 
            stimulus, 
            time_window=(-50, 200), 
        ):
        """"""

        sample_window = np.array(time_window) * 30
        num_samples = int(sample_window[1] - sample_window[0])
        spikes = np.zeros((len(stimulus), num_samples))
        for idx, event in enumerate(stimulus):
            window = (sample_window + event.sample_onset).astype(int)
            spk = self.get_spike_times(sample_window=window)
            spikes[idx, :] = spk
        return spikes


    def get_waveforms(
            self, 
        ):
        """"""

        # First, we load the TemplateModel.
        model = load_model(self._phy_path)  # first argument: path to params.py
        # We obtain the cluster id from the command-line arguments.
        cluster_id = int(self.ID)  # second argument: cluster index
        # We get the waveforms of the cluster.
        waveforms = model.get_cluster_spike_waveforms(cluster_id)
        n_spikes, n_samples, n_channels_loc = waveforms.shape
        # We get the channel ids where the waveforms are located.
        channel_ids = model.get_cluster_channels(cluster_id)
        # Get the waveforms from the peak channel.
        try:
            (peak_idx,) = np.where(channel_ids == self.peak_channel)[0]
            return waveforms[:, :, peak_idx]
        except:
            return None


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
            drop_idx=[], 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Population.spike_times.tocsr()
            (unit_idx,) = np.where(self._Population.units["unit_id"] == self.ID)[0]
            self.spike_times = spike_times_matrix[unit_idx].tocsc()
            if len(drop_idx) > 0:
                mask = np.array([True] * spike_times.size)
                mask[drop_idx.astype(int)] = False
                spike_times = self.spike_times[mask]
            else:
                spike_times = self.spike_times
        else:
            spike_times = self.spike_times
        if (sample_window[0] is None) & (sample_window[1] is None):
            return spike_times[0, :].toarray().squeeze()
        else:
            return spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()


    def bin_spikes(
            self, 
            bin_size=100, 
            type="rate", 
            drop_idx=[], 
        ):
        """"""
        
        drop_end = int((self.total_samples - len(drop_idx)) % (bin_size * 30))
        num_bins = int((self.total_samples - len(drop_idx) - drop_end)/(bin_size * 30))
        spike_times = self.get_spike_times()
        if len(drop_idx) > 0:
            mask = np.array([True] * spike_times.size)
            mask[drop_idx.astype(int)] = False
            spike_times = spike_times[mask]
        spike_times = spike_times[:-drop_end]
        if type == "rate":
            spikes = spike_times.reshape((num_bins, -1)).sum(axis=1) / (bin_size / 1000)
        elif type == "count":
            spikes = spike_times.reshape((num_bins, -1)).sum(axis=1)
        return spikes

