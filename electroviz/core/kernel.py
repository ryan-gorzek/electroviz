
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
import math

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
            resp = unit.get_spike_times(sample_window=window)
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


    def _rebase_norms(
            self, 
            kernels, 
            norms, 
        ):
        """"""

        base_norms = np.array([np.linalg.norm(kern.flatten()) for kern in kernels])
        if not all(base_norms == 0):
            (idx,) = np.where(base_norms > 0)
            new_norms = np.empty(norms.shape)
            for norm_idx, norm in enumerate(norms):
                new_norms[norm_idx] = (np.linalg.norm(kernels[norm_idx].flatten()) / np.linalg.norm(kernels[idx[0]].flatten())) ** 2
        else:
            new_norms = norms
        return new_norms




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
        kerns, norms = self._compute_kernels(self.response_windows)
        self.ONs, self.OFFs = kerns
        self.ON_norms, self.OFF_norms = norms


    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
            ax_in=None, 
            type="peak", 
            return_t=False, 
        ):
        """"""

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, axs = plt.subplots(3, 1)
        else:
            axs = ax_in
        
        if not all((np.isinf(self.ON_norms) | np.isnan(self.ON_norms)) |
                   (np.isinf(self.OFF_norms) | np.isnan(self.OFF_norms))):
            if type == "peak":
                (ON_t,) = np.where(self.ON_norms == self.ON_norms.max())
                (OFF_t,) = np.where(self.OFF_norms == self.OFF_norms.max())
            elif type == "valley":
                (ON_t,) = np.where(self.ON_norms == self.ON_norms.min())
                (OFF_t,) = np.where(self.OFF_norms == self.OFF_norms.min())
            ON = self.ONs[ON_t[0], :, :].squeeze()
            OFF = self.OFFs[ON_t[0], :, :].squeeze()
            DIFF = ON - OFF
            clim = [np.minimum(ON.min(), OFF.min()), np.maximum(ON.max(), OFF.max())]
            clim_diff = [DIFF.min(), DIFF.max()]
        else:
            ON_t, OFF_t = None, None
            ON = np.tile(0.25, self.ONs[0].shape)
            OFF = np.tile(0.25, self.OFFs[0].shape)
            DIFF = np.tile(0.25, self.ONs[0].shape)
            cmap = "binary"
            clim, clim_diff = (0, 1), (0, 1)
        axs[0].matshow(ON, cmap=cmap, clim=clim)
        axs[0].xaxis.tick_bottom()
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].matshow(OFF, cmap=cmap, clim=clim)
        axs[1].xaxis.tick_bottom()
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].matshow(DIFF, cmap=cmap, clim=clim_diff)
        axs[2].xaxis.tick_bottom()
        axs[2].axis("off")
        axs[2].set_title("ON - OFF", fontsize=18)
        if ax_in is None:
            plt.show(block=False)
            fig.set_size_inches(4.5, 10)
        if return_t is True:
            return (ON_t[0], OFF_t[0])


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
            ax_in=None, 
        ):
        """"""

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
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
        ends = np.array((np.nanmin(norms), np.nanmax(norms)))
        rng = np.diff(ends)
        lims = ends + 0.1 * np.array((-rng, rng)).T
        if lims[0][0] < 0:
            lims[0][0] = 0
        try:
            ax.set_ylim(lims[0])
        except:
            ax.set_ylim([0, 1])
        if ax_in is None:
            plt.show(block=False)
            plt.tight_layout()
            fig.set_size_inches(6, 6)


    def _compute_kernels(
            self, 
            response_windows, 
        ):
        """"""

        ONs, OFFs = (np.empty((len(response_windows), *self._Stimulus.shape[2:0:-1])) for k in range(2))
        ON_norms, OFF_norms = (np.empty((len(response_windows),)) for k in range(2))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            kernels = np.zeros(self._Stimulus.shape[:3])
            for stim_indices in np.ndindex(self._Stimulus.shape[:3]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :].mean(axis=(0, 1))
                kernels[*stim_indices] += resp_rate
            ONs[idx] = kernels[1].T[::-1, :]
            ON_norms[idx] = (np.linalg.norm(ONs[idx].flatten()) / np.linalg.norm(ONs[0].flatten())) ** 2
            OFFs[idx] = kernels[0].T[::-1, :]
            OFF_norms[idx] = (np.linalg.norm(OFFs[idx].flatten()) / np.linalg.norm(OFFs[0].flatten())) ** 2
        ON_norms = self._rebase_norms(ONs, ON_norms)
        OFF_norms = self._rebase_norms(OFFs, OFF_norms)
        return (ONs, OFFs), (ON_norms, OFF_norms)




class StaticGratingsKernel(Kernel):
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            avg_gratings=False, 
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
        self.kerns, self.norms = self._compute_kernels(self.response_windows)
        if avg_gratings is True:
            self.avg_gratings = self._compute_average_gratings(self.response_windows, self.norms)


    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
            ax_in=None, 
            type="peak", 
            return_t=False, 
        ):
        """"""

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
        if not all((np.isinf(self.norms) | np.isnan(self.norms))):
            if type == "peak":
                (t,) = np.where(self.norms == self.norms.max())
            elif type == "valley":
                (t,) = np.where(self.norms == self.norms.min())
            orisf = self.kerns[t[0], :, :].squeeze()
            clim = [orisf.min(), orisf.max()]
        else:
            t = None
            orisf = np.tile(0.25, self.kerns[0].shape)
            cmap = "binary"
            clim = (0, 1)
        ax.matshow(orisf, cmap=cmap, clim=clim)
        ax.xaxis.tick_bottom()
        oris = np.unique(np.array(self._Stimulus.unique)[:, 0])
        sfs = np.unique(np.array(self._Stimulus.unique)[:, 1])
        ax.set_xticks(np.arange(1, len(sfs), 2))
        ax.set_xticklabels(np.round(sfs[1::2], decimals=3))
        ax.set_xlabel("Spatial Frequency")
        ax.set_yticks(np.arange(0, len(oris), 2))
        ax.set_yticklabels(oris[-1::-2].astype(int))
        ax.set_ylabel("Orientation")
        if ax_in is None:
            plt.show(block=False)
            fig.subplots_adjust(left=0.15, bottom=0.11, right=0.95, top=0.95)
            fig.set_size_inches(6, 6)
        if return_t is True:
            if t is not None:
                return t[0]
            else:
                return t


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
            ax_in=None, 
        ):
        """"""

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
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
        if lims[0][0] < 0:
            lims[0][0] = 0
        ax.set_ylim(lims[0])
        if ax_in is None:
            plt.show(block=False)
            plt.tight_layout()
            fig.set_size_inches(6, 6)



    def plot_average_gratings(
            self, 
            ax_in=None, 
        ):
        """"""

        if ax_in is None:
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
        else:
            ax = ax_in
        ax.imshow(self.avg_gratings)
        ax.axis("off")
        plt.show(block=False)

    def _compute_kernels(
            self, 
            response_windows, 
        ):
        """"""

        orisfs = np.empty((len(response_windows), *self._Stimulus.shape[:2]))
        orisf_norms = np.empty((len(response_windows),))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            kernels = np.zeros(self._Stimulus.shape[:2])
            for stim_indices in np.ndindex(self._Stimulus.shape[:2]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :, :].mean(axis=(0, 1, 2))
                kernels[*stim_indices] += resp_rate
            orisfs[idx] = kernels[::-1, :]
            orisf_norms[idx] = (np.linalg.norm(orisfs[idx].flatten()) / np.linalg.norm(orisfs[0].flatten())) ** 2
        orisf_norms = self._rebase_norms(orisfs, orisf_norms)
        return orisfs, orisf_norms


    def _compute_average_gratings(
            self, 
            response_windows, 
            norms, 
        ):
        """"""

        gratings = self._Stimulus.gratings
        gratings_mult = []
        if not all((np.isinf(self.norms) | np.isnan(self.norms))):
            (t,) = np.where(self.norms == self.norms.max())
            resp_on, resp_off = self._time_to_bins(response_windows[t[0]])
            for idx, stim_indices in enumerate(np.ndindex(self._Stimulus.shape[:2])):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :, :].mean(axis=(0, 1, 2))
                mult = gratings[idx, :, :] * resp_rate 
                gratings_mult.append(mult)
        return np.array(gratings_mult).mean(axis=0)

