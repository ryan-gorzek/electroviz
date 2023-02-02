
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from electroviz.viz.raster import RateRaster
import copy
from scipy.signal import butter, lfilter
from electroviz.utils.icsd import compute_csd
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from scipy.stats import zscore

class Probe:
    """
    
    """


    def __init__(
            self, 
            lfp, 
            sync, 
        ):
        """"""

        self._Sync = sync
        self.channels = lfp.channels[::-1, :]
        self.channel_positions = lfp.channel_positions
        self.sampling_rate = lfp.sampling_rate
        self.total_samples = lfp.total_samples
        self.total_time = lfp.total_samples / lfp.sampling_rate
        self.total_channels = self.channels.shape[0]
        # Define current index for iteration.
        self._current_channel_idx = 0

    def __getitem__(
            self, 
            input, 
        ):
        """"""

        try:
            subset = self.channels[input, :]
        except TypeError:
            print("Failed.")
        return subset


    def __iter__(self):
        """"""

        return iter(self.channels)


    def __next__(self):
        """"""

        if self._current_channel_idx < self.total_channels:
            channel = self.channels[self._current_channel_idx, :]
            self._current_channel_idx += 1
            return channel


    def __len__(self):
        """"""

        return self.channels.shape[0]


    def plot_CSD(
            self, 
            stimulus, 
            subtract=False, 
        ):
        """"""

        chan_mask = self.filter_channels(x=43).channel_positions[:, 1] <= 1000
        channels = self.filter_channels(x=43)._filter_freq().get_response(stimulus, time_window=(-50, 100))
        chans_in = channels[chan_mask, :]
        if subtract is True:
            chans_in = chans_in[:, 50:].T - chans_in[:, :50].mean(axis=1)
        else:
            chans_in = chans_in[:, 50:].T
        contacts = np.linspace(0, 1000E-6, 25)*pq.m
        csd = compute_csd(chans_in.T, coord_electrodes=contacts, mode="exp", gauss_filter=(1.4, 0), diam=800E-6 * pq.m)

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        if subtract is True:
            ax.imshow(csd, cmap="RdBu_r", aspect=5, clim=(-350, 350))
        else:
            ax.imshow(zscore(np.array(csd), axis=1), cmap="RdBu_r", aspect=5)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xticklabels(np.arange(0, 101, 10))
        ax.set_yticks(np.arange(0, 25, 5))
        ax.set_yticklabels(np.arange(0, 1000, 200))
        plt.show(block=False)


    def plot_power(
            self, 
        ):
        """"""

        chan_mask = self.filter_channels(x=43).channel_positions[:, 1] <= 1000
        channels = self.filter_channels(x=43).channels[chan_mask, :]
        powers = np.zeros((channels.shape[0],))
        for chan, power in zip(channels, powers):
            ps = np.abs(np.fft.fft(chan))**2
            time_step = 1 / 2500
            freqs = np.fft.fftfreq(chan.size, time_step)
            idx = np.argsort(freqs)
            cut = int((idx.size / 2500) * 750)
            power = ps[idx][:cut].mean()
        
        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        ax.plot(powers, np.arange(0, powers.size, 1))
        plt.show(block=False)


    def plot_channels(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            yscale="Depth", 
            fig_size=(6, 9), 
            save_path="", 
        ):
        """"""

        responses = self.get_response(stimulus, time_window)

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        for idx, resp in enumerate(responses[::-2]):
            ax.plot(range(*time_window), zscore(resp) + idx, color="k")
        plt.show(block=False)


    def plot_raster(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            yscale="Depth", 
            fig_size=(6, 9), 
            save_path="", 
            ax_in=None, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window)
        if yscale == "Depth":
            depths = self.channel_positions[:, 1].astype(int)
            RateRaster(time_window, responses, ylabel="Depth", cmap="RdBu_r", yscale=depths, fig_size=fig_size, save_path=save_path, ax_in=ax_in)
        elif yscale == "Channel":
            RateRaster(time_window, responses, ylabel="Channel", fig_size=fig_size, save_path=save_path, ax_in=ax_in)


    def filter_channels(
            self, 
            x=None, 
            y=None, 
        ):
        """"""

        if (x is not None) & (y is not None):
            mask = np.logical_and(self.channel_positions[:, 0] == x, self.channel_positions[:, 1] == y)
        if x is not None:
            mask = self.channel_positions[:, 0] == x
        elif y is not None:
            mask = self.channel_positions[:, 1] == y

        subset = copy.copy(self)
        subset.channel_positions = self.channel_positions[mask, :]
        subset.total_channels = np.nonzero(mask)
        subset.channels = self.channels[mask, :]
        return subset


    def get_response(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
        ):
        """"""
        
        stimulus = stimulus.lfp()
        sample_window = np.array(time_window) * 2.5
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(time_window[1] - time_window[0])
        responses = np.zeros((len(self), num_bins, len(stimulus)))
        for idx, event in enumerate(stimulus):
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.channels[:, window[0]:window[1]]
            bin_idx = 0
            for bin_pair in np.arange(0, num_samples, 5, dtype=int):
                middle = resp[:, bin_pair + 2]  / 2
                first_bin = (resp[:, bin_pair : bin_pair+1].mean(axis=1) + middle) / 2.5
                second_bin = (resp[:, bin_pair+3 : bin_pair+5].mean(axis=1) + middle) / 2.5
                responses[:, bin_idx, idx] = first_bin
                responses[:, bin_idx+1, idx] = second_bin
                bin_idx += 2
        return responses.mean(axis=2)


    def _filter_freq(
            self, 
        ):
        """"""

        subset = copy.copy(self)
        b, a = butter(5, 500, fs=2500, btype="low")
        subset.channels = lfilter(b, a, self.channels)
        return subset

