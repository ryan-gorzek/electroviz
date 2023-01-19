
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
from scipy.stats import zscore
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from electroviz.viz.psth import PSTH

class Kernel:
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
        ):
        """"""

        self._Stimulus = stimulus
        self.time_window = time_window
        self.bin_size = bin_size
        self._total_time = (time_window[1] + np.abs(time_window[0]))
        sample_window = np.array(time_window) * 30
        num_samples = int(sample_window[1] - sample_window[0])
        self._num_bins = int(num_samples/(bin_size * 30))
        self._responses = np.zeros((self._num_bins, *stimulus.shape))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = unit.get_spike_times(window)
            spikes_per_sec = resp.reshape(self._num_bins, -1).sum(axis=1) / (bin_size / 1000)
            self._responses[:, *event.stim_indices] = spikes_per_sec


    def _time_to_bins(
            self, 
            window, 
        ):
        """"""

        time_window_bins = np.linspace(*self.time_window, self._num_bins)
        on_dists = np.abs(time_window_bins - window[0])
        (on_idx,) = np.where(on_dists == np.min(on_dists))
        off_dists = np.abs(time_window_bins - window[1])
        (off_idx,) = np.where(off_dists == np.min(off_dists))
        return int(on_idx[0]), int(off_idx[0] + 1)




class SparseNoiseKernel(Kernel):
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        sample_window = np.array(time_window) * (unit.sampling_rate / 1000)
        num_samples = int(sample_window[1] - sample_window[0])
        self.response_windows = [(on, on + bin_size) for on in np.arange(*self.time_window, self.bin_size)]
        kern, tmax, norm, _ = self._compute_kernels(self.response_windows)
        self.ON, self.OFF, self.DIFF = kern
        self.ON_tmax, self.OFF_tmax = tmax
        self.ON_S, self.OFF_S = norm
        # Initialize fits as None.
        self.ON_fit, self.OFF_fit, self.DIFF_fit = None, None, None


    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, 1)
        axs[0].matshow(self.ON, cmap=cmap, clim=[self.ON.min(), self.ON.max()])
        axs[0].xaxis.tick_bottom()
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].matshow(self.OFF, cmap=cmap, clim=[self.OFF.min(), self.OFF.max()])
        axs[1].xaxis.tick_bottom()
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].matshow(self.DIFF, cmap=cmap, clim=[self.DIFF.min(), self.DIFF.max()])
        axs[2].xaxis.tick_bottom()
        axs[2].axis("off")
        axs[2].set_title("ON - OFF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(4.5, 10)


    def plot_raw_delay(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, len(self.response_windows))
        _, (ON_tmax, OFF_tmax), _, (ON_all, OFF_all) = self._compute_kernels(self.response_windows)
        for idx, (ON, OFF, window) in enumerate(zip(ON_all, OFF_all, self.response_windows)):
            DIFF = ON - OFF
            axs[0][idx].matshow(ON, cmap=cmap, clim=[ON.min(), ON.max()])
            axs[0][idx].xaxis.tick_bottom()
            axs[0][idx].axis("off")
            axs[0][idx].set_title(window[0])
            axs[1][idx].matshow(OFF, cmap=cmap, clim=[OFF.min(), OFF.max()])
            axs[1][idx].xaxis.tick_bottom()
            axs[1][idx].axis("off")
            axs[2][idx].matshow(DIFF, cmap=cmap, clim=[DIFF.min(), DIFF.max()])
            axs[2][idx].xaxis.tick_bottom()
            axs[2][idx].axis("off")
            if ON_tmax[0] == idx:
                axs[0][idx].text(0.5, -0.35, "*", fontsize=20, 
                                                  fontweight="bold", 
                                                  horizontalalignment="center",
                                                  verticalalignment="center", 
                                                  transform=axs[0][idx].transAxes)
            if OFF_tmax[0] == idx:
                axs[1][idx].text(0.5, -0.35, "*", fontsize=20, 
                                                  fontweight="bold", 
                                                  horizontalalignment="center",
                                                  verticalalignment="center", 
                                                  transform=axs[1][idx].transAxes)
        
        axs[0][0].text(-0.2, 0.5, "ON", horizontalalignment="center",
                                        verticalalignment="center", 
                                        rotation=90, 
                                        transform=axs[0][0].transAxes)
        axs[1][0].text(-0.2, 0.5, "OFF", horizontalalignment="center",
                                         verticalalignment="center", 
                                         rotation=90, 
                                         transform=axs[1][0].transAxes)
        axs[2][0].text(-0.2, 0.5, "ON - OFF", horizontalalignment="center",
                                              verticalalignment="center", 
                                              rotation=90, 
                                              transform=axs[2][0].transAxes)
        plt.show(block=False)
        fig.subplots_adjust(left=0.01, right=0.99)
        fig.set_size_inches(len(self.response_windows), 3)


    def plot_norm_delay(
            self, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        t = np.linspace(0, self.ON_S.size, self.ON_S.size)
        ax.hlines(1, -1, t.size + 1, colors=(0.0, 0.0, 0.0, 0.75), linestyles="--")
        ax.bar(t, self.ON_S, color=(0.9, 0.2, 0.2, 0.5), label="ON")
        ax.bar(t, self.OFF_S, color=(0.2, 0.2, 0.9, 0.5), label="OFF")
        ax.legend(frameon=False)
        ax.set_xticks(np.linspace(0, t.size, 6))
        ax.set_xticklabels(np.linspace(*self.time_window, 6))
        ax.set_xlabel("Time from Onset (ms)", fontsize=16)
        ax.set_ylabel("|| Kernel(t) || / || Kernel(0) ||", fontsize=16)
        norms = np.concatenate((self.ON_S, self.OFF_S))
        ends = np.array((norms.min(), norms.max()))
        rng = np.diff(ends)
        lims = ends + 0.1 * np.array((-rng, rng)).T
        ax.set_ylim(lims[0])
        plt.show(block=False)
        plt.tight_layout()
        fig.set_size_inches(6, 6)


    def _compute_kernels(
            self, 
            response_windows, 
        ):
        """"""

        ONs, OFFs = (np.empty((len(response_windows), *self._Stimulus.shape[2:0:-1])) for k in range(2))
        ON_S, OFF_S = (np.empty((len(response_windows),)) for k in range(2))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            kernels = np.zeros(self._Stimulus.shape[:3])
            for stim_indices in np.ndindex(self._Stimulus.shape[:3]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :].mean(axis=(0, 1))
                kernels[*stim_indices] += resp_rate
            ONs[idx] = kernels[1].T[::-1, :]
            ON_S[idx] = np.linalg.norm(ONs[idx].flatten()) / np.linalg.norm(ONs[0].flatten())
            OFFs[idx] = kernels[0].T[::-1, :]
            OFF_S[idx] = np.linalg.norm(OFFs[idx].flatten()) / np.linalg.norm(OFFs[0].flatten())
        # Get the kernels with the maximum norm (across time).
        if not all((np.isinf(ON_S) | np.isnan(ON_S)) |
                   (np.isinf(OFF_S) | np.isnan(OFF_S))):
            (ON_tmax,) = np.where(ON_S == np.max(ON_S))
            ON_opt = ONs[ON_tmax[0], :, :].squeeze()
            (OFF_tmax,) = np.where(OFF_S == np.max(OFF_S))
            OFF_opt = OFFs[OFF_tmax[0], :, :].squeeze()
            DIFF_opt = ON_opt - OFF_opt
        else:
            ON_tmax, OFF_tmax = np.nan, np.nan
            ON_opt = np.empty(ONs.shape[1:3]).fill(np.nan)
            OFF_opt = np.empty(OFFs.shape[1:3]).fill(np.nan)
            DIFF_opt = np.empty(ONs.shape[1:3]).fill(np.nan)
        return (ON_opt, OFF_opt, DIFF_opt), (ON_tmax, OFF_tmax), (ON_S, OFF_S), (ONs, OFFs)




class StaticGratingsKernel(Kernel):
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        sample_window = np.array(time_window) * (unit.sampling_rate / 1000)
        num_samples = int(sample_window[1] - sample_window[0])
        self.response_windows = [(on, on + bin_size) for on in np.arange(*self.time_window, self.bin_size)]
        self.orisf, self.orisf_tmax, self.orisf_S, _ = self._compute_kernels(self.response_windows)
        # Initialize fits as None.
        self.orisf_fit = None


    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        ax.matshow(self.orisf, cmap=cmap, clim=[self.orisf.min(), self.orisf.max()])
        ax.xaxis.tick_bottom()
        oris = np.unique(np.array(self._Stimulus.unique)[:, 0])
        sfs = np.unique(np.array(self._Stimulus.unique)[:, 1])
        ax.set_xticks(np.arange(0, len(sfs), 2))
        ax.set_xticklabels(sfs[0::2])
        ax.set_xlabel("Spatial Frequency")
        ax.set_yticks(np.arange(0, len(oris), 2))
        ax.set_yticklabels(oris[-1::-2].astype(int))
        ax.set_ylabel("Orientation")
        plt.show(block=False)
        fig.subplots_adjust(left=0.15, bottom=0.11, right=0.95, top=0.95)
        fig.set_size_inches(6, 6)


    def plot_raw_delay(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(1, len(self.response_windows))
        _, orisf_tmax, _, orisf_all = self._compute_kernels(self.response_windows)
        for idx, (orisf, window) in enumerate(zip(orisf_all, self.response_windows)):
            axs[idx].matshow(orisf, cmap=cmap, clim=[orisf.min(), orisf.max()])
            axs[idx].xaxis.tick_bottom()
            axs[idx].axis("off")
            axs[idx].set_title(window[0])
            if orisf_tmax[0] == idx:
                axs[idx].text(0.5, -0.35, "*", fontsize=20, 
                                               fontweight="bold", 
                                               horizontalalignment="center",
                                               verticalalignment="center", 
                                               transform=axs[idx].transAxes)
            if idx == 0:
                axs[0].set_xlabel("Spatial Frequency", fontsize=14)
                axs[0].set_ylabel("Orientation", fontsize=14)
        plt.show(block=False)
        fig.subplots_adjust(left=0.01, right=0.99)
        fig.set_size_inches(len(self.response_windows), 3)


    def plot_norm_delay(
            self, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        t = np.linspace(0, self.orisf_S.size, self.orisf_S.size)
        ax.hlines(1, -1, t.size + 1, colors=(0.0, 0.0, 0.0, 0.75), linestyles="--")
        ax.bar(t, self.orisf_S, color=(0, 0, 0, 0.5))
        ax.set_xticks(np.linspace(0, t.size, 6))
        ax.set_xticklabels(np.linspace(*self.time_window, 6))
        ax.set_xlabel("Time from Onset (ms)", fontsize=16)
        ax.set_ylabel("|| Kernel(t) || / || Kernel(0) ||", fontsize=16)
        ends = np.array((self.orisf_S.min(), self.orisf_S.max()))
        rng = np.diff(ends)
        lims = ends + 0.1 * np.array((-rng, rng)).T
        ax.set_ylim(lims[0])
        plt.show(block=False)
        plt.tight_layout()
        fig.set_size_inches(6, 6)


    def _compute_kernels(
            self, 
            response_windows, 
        ):
        """"""

        orisfs = np.empty((len(response_windows), *self._Stimulus.shape[:2]))
        orisf_S = np.empty((len(response_windows),))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            kernels = np.zeros(self._Stimulus.shape[:2])
            for stim_indices in np.ndindex(self._Stimulus.shape[:2]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :, :].mean(axis=(0, 1, 2))
                kernels[*stim_indices] += resp_rate
            orisfs[idx] = kernels[::-1, :]
            orisf_S[idx] = np.linalg.norm(orisfs[idx].flatten()) / np.linalg.norm(orisfs[0].flatten())
        # Get the kernels with the maximum norm (across time).
        if not all((np.isinf(orisf_S) | np.isnan(orisf_S))):
            (orisf_tmax,) = np.where(orisf_S == np.max(orisf_S))
            orisf_opt = orisfs[orisf_tmax[0], ::-1, :].squeeze()
        else:
            orisf_tmax = np.nan
            orisf_opt = np.empty(orisfs.shape[:2]).fill(np.nan)
        return orisf_opt, orisf_tmax, orisf_S, orisfs

