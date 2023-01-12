# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
from scipy.stats import zscore
from scipy.optimize import curve_fit


class Kernel:
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.005, 
        ):
        """"""

        self.total_time = (time_window[1] + np.abs(time_window[0]))*1000
        self.time_window = time_window
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        self.num_bins = int(num_samples/(bin_size*30000))
        self.responses = np.zeros((self.num_bins, *stimulus.shape))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = unit.get_spike_times(window)
            spikes_per_sec = np.sum(resp.reshape(self.num_bins, -1), axis=1) / bin_size
            self.responses[:, *event.stim_indices] = spikes_per_sec

    def _time_to_bins(
            self, 
            window, 
            time_window, 
            num_bins, 
        ):
        """"""

        time_window_bins = np.linspace(*time_window, num_bins)
        on_dists = np.abs(time_window_bins - window[0])
        (on_idx,) = np.where(on_dists == np.min(on_dists))
        off_dists = np.abs(time_window_bins - window[1])
        (off_idx,) = np.where(off_dists == np.min(off_dists))
        return int(on_idx), int(off_idx + 1)



class SparseNoiseKernel(Kernel):
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.005, 
            resp_window=[0.030, 0.050], 
            base_window=[-0.030, -0.010], 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        self._Stimulus = stimulus
        kernels = np.zeros(stimulus.shape[0:3])
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        self.num_bins = int(num_samples/(bin_size*30000))
        self.resp_window, self.base_window = resp_window, base_window
        resp_on, resp_off = self._time_to_bins(resp_window, time_window, self.num_bins)
        base_on, base_off = self._time_to_bins(base_window, time_window, self.num_bins)
        for stim_indices in np.ndindex(stimulus.shape[:3]):
            resp_count = self.responses[resp_on:resp_off, *stim_indices, :].mean(axis=(0, 1))
            base_count = self.responses[base_on:base_off, *stim_indices, :].mean(axis=(0, 1))
            kernels[*stim_indices] += (resp_count - base_count)
        self._kernels = kernels
        self.OFF = kernels[0]
        self.ON = kernels[1]
        self.DIFF = self.ON - self.OFF
        # Initialize fits as None.
        self.ON_fit, self.OFF_fit, self.DIFF_fit = None, None, None

    # def fit(
    #         self, 
    #     ):
    #     """"""
        
        

    #     def gaussian_2D():


    def get_response(
            self, 
            base_norm=True, 
        ):
        """"""

        resp_on, resp_off = self._time_to_bins(self.resp_window, self.time_window, self.num_bins)
        base_on, base_off = self._time_to_bins(self.base_window, self.time_window, self.num_bins)
        response_mean = self.responses.mean(axis=(1,2,3,4))
        if base_norm == True:
            response = response_mean[resp_on:resp_off].mean() - response_mean[base_on:base_off].mean()
        else:
            response = response_mean[resp_on:resp_off].mean()
        return response

    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(self.ON.T, cmap=cmap, clim=[self.ON.min(axis=(0, 1)), self.ON.max(axis=(0, 1))])
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].imshow(self.OFF.T, cmap=cmap, clim=[self.OFF.min(axis=(0, 1)), self.OFF.max(axis=(0, 1))])
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].imshow(self.DIFF.T, cmap=cmap, clim=[self.DIFF.min(axis=(0, 1)), self.DIFF.max(axis=(0, 1))])
        axs[2].axis("off")
        axs[2].set_title("ON - OFF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(4, 8)

    def plot_fit(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(self.ON_fit.T, cmap=cmap, clim=[self.ON_fit.min(axis=(0, 1)), self.ON_fit.max(axis=(0, 1))])
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].imshow(self.OFF_fit.T, cmap=cmap, clim=[self.OFF_fit.min(axis=(0, 1)), self.OFF_fit.max(axis=(0, 1))])
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].imshow(self.DIFF_fit.T, cmap=cmap, clim=[self.DIFF_fit.min(axis=(0, 1)), self.DIFF_fit.max(axis=(0, 1))])
        axs[2].axis("off")
        axs[2].set_title("ON - OFF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(4, 8)

    def plot_PETH(
            self, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        response = self.responses.mean(axis=(1,2,3,4))
        axs.bar(range(self.num_bins), response, color="k")
        axs.set_xlabel("Time from onset (ms)", fontsize=16)
        axs.set_xticks(np.linspace(0, self.num_bins, 6))
        axs.set_xticklabels(np.linspace(self.time_window[0]*1000, self.time_window[1]*1000, 6))
        axs.set_ylabel("Spikes/s", fontsize=16)
        plt.show(block=False)
        fig.set_size_inches(4, 4)


class StaticGratingsKernel(Kernel):
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.005, 
            resp_window=[0.030, 0.060], 
            base_window=[-0.040, -0.010], 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        self._Stimulus = stimulus
        kernels = np.zeros(stimulus.shape[0:2])
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        self.num_bins = int(num_samples/(bin_size*30000))
        self.resp_window, self.base_window = resp_window, base_window
        resp_on, resp_off = self._time_to_bins(resp_window, time_window, self.num_bins)
        base_on, base_off = self._time_to_bins(base_window, time_window, self.num_bins)
        for stim_indices in np.ndindex(stimulus.shape[:2]):
            resp_count = self.responses[resp_on:resp_off, *stim_indices, :, :].mean(axis=(0, 1, 2))
            base_count = self.responses[base_on:base_off, *stim_indices, :, :].mean(axis=(0, 1, 2))
            kernels[*stim_indices] += (resp_count - base_count)
        self.kernel = kernels

    # def _fit_2D_gaussian(
    #         self, 
    #     ):
    #     """"""

    def get_response(
            self, 
            base_norm=True, 
        ):
        """"""

        resp_on, resp_off = self._time_to_bins(self.resp_window, self.time_window, self.num_bins)
        base_on, base_off = self._time_to_bins(self.base_window, self.time_window, self.num_bins)
        response_mean = self.responses.mean(axis=(1,2,3,4))
        if base_norm == True:
            response = response_mean[resp_on:resp_off].mean() - response_mean[base_on:base_off].mean()
        else:
            response = response_mean[resp_on:resp_off].mean()
        return response

    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        axs.imshow(self.kernel, cmap=cmap, clim=[self.kernel.min(axis=(0, 1)), self.kernel.max(axis=(0, 1))])
        axs.axis("off")
        axs.set_title("Ori/SF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(4, 4)

    def plot_PETH(
            self, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots()
        response = self.responses.mean(axis=(1,2,3,4))
        axs.bar(range(self.num_bins), response, color="k")
        axs.set_xlabel("Time from onset (ms)", fontsize=16)
        axs.set_xticks(np.linspace(0, self.num_bins, 6))
        axs.set_xticklabels(np.linspace(self.time_window[0]*1000, self.time_window[1]*1000, 6))
        axs.set_ylabel("Spikes/s", fontsize=16)
        plt.show(block=False)
        fig.set_size_inches(4, 4)
