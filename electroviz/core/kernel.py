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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Kernel:
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=(-0.050, 0.200), 
            bin_size=0.005, 
        ):
        """"""

        self._Stimulus = stimulus
        self.time_window = time_window
        self.bin_size = bin_size
        self._total_time = (time_window[1] + np.abs(time_window[0]))*1000
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        self._num_bins = int(num_samples/(bin_size*30000))
        self._responses = np.zeros((self._num_bins, *stimulus.shape))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = unit.get_spike_times(window)
            spikes_per_sec = np.sum(resp.reshape(self._num_bins, -1), axis=1) / bin_size
            self._responses[:, *event.stim_indices] = spikes_per_sec

    def get_response(
            self, 
            response_window=(0.030, 0.060), 
            baseline_window=(-0.040, -0.010), 
            baseline_norm=True,  
        ):
        """"""

        resp_on, resp_off = self._time_to_bins(response_window)
        base_on, base_off = self._time_to_bins(baseline_window)
        response_mean = self._responses.mean(axis=(1,2,3,4))
        if baseline_norm == True:
            response = response_mean[resp_on:resp_off].mean() - response_mean[base_on:base_off].mean()
        else:
            response = response_mean[resp_on:resp_off].mean()
        return response

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
            time_window=(-0.050, 0.200), 
            bin_size=0.005, 
            response_window=(0.000, 0.100), 
            baseline_window=(-0.040, -0.010), 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        response_windows = [(on, on + bin_size) for on in np.arange(*response_window, self.bin_size)]
        baseline_window = baseline_window
        self.compute_kernels(response_windows, baseline_window)
        # Initialize fits as None.
        self.ON_fit, self.OFF_fit, self.DIFF_fit = None, None, None

    def compute_kernels(
            self, 
            response_windows, 
            baseline_window, 
        ):
        """"""

        self.response_windows = response_windows
        self.baseline_window = baseline_window
        ON, OFF = (np.empty((len(response_windows), *self._Stimulus.shape[2:0:-1])) for k in range(2))
        self.ON_S, self.OFF_S = (np.empty((len(response_windows),)) for k in range(2))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            base_on, base_off = self._time_to_bins(baseline_window)
            kernels = np.zeros(self._Stimulus.shape[0:3])
            for stim_indices in np.ndindex(self._Stimulus.shape[:3]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :].mean(axis=(0, 1))
                base_rate = self._responses[base_on:base_off, *stim_indices, :].mean(axis=(0, 1))
                kernels[*stim_indices] += (resp_rate - base_rate)
            ON[idx] = kernels[1].T
            self.ON_S[idx] = np.linalg.norm(ON[idx].flatten()) / np.linalg.norm(ON[0].flatten())
            OFF[idx] = kernels[0].T
            self.OFF_S[idx] = np.linalg.norm(OFF[idx].flatten()) / np.linalg.norm(OFF[0].flatten())
        # Get the kernels with the maximum norm (across time).
        if not all((np.isinf(self.ON_S) | np.isnan(self.ON_S)) |
                   (np.isinf(self.OFF_S) | np.isnan(self.OFF_S))):
            (ON_tmax,) = np.where(self.ON_S == np.max(self.ON_S))
            self.ON = ON[ON_tmax[0], :, :].squeeze()
            (OFF_tmax,) = np.where(self.OFF_S == np.max(self.OFF_S))
            self.OFF = OFF[OFF_tmax[0], :, :].squeeze()
            self.DIFF = self.ON - self.OFF
        else:
            self.ON = np.empty(ON.shape[1:3]).fill(np.nan)
            self.OFF = np.empty(OFF.shape[1:3]).fill(np.nan)
            self.DIFF = np.empty(ON.shape[1:3]).fill(np.nan)

    def get_norm(
            self, 
        ):
        """"""

        norms = (np.max(self.ON_S), np.max(self.OFF_S))
        return np.max(norms)

    def plot_raw(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(self.ON, cmap=cmap, clim=[self.ON.min(axis=(0, 1)), self.ON.max(axis=(0, 1))])
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].imshow(self.OFF, cmap=cmap, clim=[self.OFF.min(axis=(0, 1)), self.OFF.max(axis=(0, 1))])
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].imshow(self.DIFF, cmap=cmap, clim=[self.DIFF.min(axis=(0, 1)), self.DIFF.max(axis=(0, 1))])
        axs[2].axis("off")
        axs[2].set_title("ON - OFF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(4, 8)

    def plot_raw_delay(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(len(self.resp_windows), 3)
        for ON, OFF in zip(self.ON, self.OFF):
            kernels = self.compute(resp_window, self.base_window)
            ON = kernels[1].T
            OFF = kernels[0].T
            DIFF = ON - OFF
            axs[idx][0].imshow(ON, cmap=cmap, clim=[ON.min(), ON.max()])
            axs[idx][0].axis("off")
            axs[idx][0].set_title("ON", fontsize=18)
            axs[idx][1].imshow(OFF, cmap=cmap, clim=[OFF.min(), OFF.max()])
            axs[idx][1].axis("off")
            axs[idx][1].set_title("OFF", fontsize=18)
            axs[idx][2].imshow(DIFF, cmap=cmap, clim=[DIFF.min(), DIFF.max()])
            axs[idx][2].axis("off")
            axs[idx][2].set_title("ON - OFF", fontsize=18)
        plt.show(block=False)
        fig.set_size_inches(8, 4 * (len(resp_windows) - 1))

    def plot_fit(
            self, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(self.ON_fit.T, cmap=cmap, clim=[self.ON_fit.min(), self.ON_fit.max()])
        axs[0].axis("off")
        axs[0].set_title("ON", fontsize=18)
        axs[1].imshow(self.OFF_fit.T, cmap=cmap, clim=[self.OFF_fit.min(), self.OFF_fit.max()])
        axs[1].axis("off")
        axs[1].set_title("OFF", fontsize=18)
        axs[2].imshow(self.DIFF_fit.T, cmap=cmap, clim=[self.DIFF_fit.min(), self.DIFF_fit.max()])
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
        response = self._responses.mean(axis=(1,2,3,4))
        axs.bar(range(self._num_bins), response, color="k")
        axs.set_xlabel("Time from onset (ms)", fontsize=16)
        axs.set_xticks(np.linspace(0, self._num_bins, 6))
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
            time_window=(-0.050, 0.200), 
            bin_size=0.005, 
            response_window=(0.000, 0.100), 
            baseline_window=(-0.040, -0.010), 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        response_windows = [(on, on + bin_size) for on in np.arange(*response_window, self.bin_size)]
        baseline_window = baseline_window
        self.compute_kernels(response_windows, baseline_window)
        # Initialize fit as None.
        self.kernel_fit = None

    def compute_kernels(
            self, 
            response_windows, 
            baseline_window, 
        ):
        """"""

        self.response_windows = response_windows
        self.baseline_window = baseline_window
        kernel = np.empty((len(response_windows), *self._Stimulus.shape[:2]))
        self.kernel_S = np.empty((len(response_windows),))
        for idx, response_window in enumerate(response_windows):
            resp_on, resp_off = self._time_to_bins(response_window)
            base_on, base_off = self._time_to_bins(baseline_window)
            kernels = np.zeros(self._Stimulus.shape[:2])
            for stim_indices in np.ndindex(self._Stimulus.shape[:2]):
                resp_rate = self._responses[resp_on:resp_off, *stim_indices, :].mean(axis=(0, 1, 2))
                base_rate = self._responses[base_on:base_off, *stim_indices, :].mean(axis=(0, 1, 2))
                kernels[*stim_indices] += (resp_rate - base_rate)
            kernel[idx] = kernels
            self.kernel_S[idx] = np.linalg.norm(kernel[idx].flatten()) / np.linalg.norm(kernel[0].flatten())
        # Get the kernel with the maximum norm (across time).
        if not all((np.isinf(self.kernel_S) | np.isnan(self.kernel_S))):
            (kernel_tmax,) = np.where(self.kernel_S == np.max(self.kernel_S))
            self.kernel = kernel[kernel_tmax[0], :, :].squeeze()
        else:
            self.kernel = np.empty(kernel.shape[:2]).fill(np.nan)

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
        # Beef this up.
        plt.show(block=False)
        fig.set_size_inches(4, 4)

    def plot_raw_delay(
            self, 
            time_window, 
            cmap="viridis", 
            save_path="", 
        ):
        """"""

        resp_windows = [(on, on + self.bin_size) for on in np.arange(*time_window, self.bin_size)]
        base_window = (self.base_window[0], self.base_window[0] + self.bin_size)
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(len(resp_windows), 1)
        for idx, resp_window in enumerate(resp_windows):
            kernel = self.compute(resp_window, base_window)
            axs[idx].imshow(kernel, cmap=cmap, clim=[kernel.min(axis=(0, 1)), kernel.max(axis=(0, 1))])
            axs[idx].axis("off")
            axs[idx].set_title("Ori/SF", fontsize=18)
            # Beef this up.
        plt.show(block=False)
        fig.set_size_inches(8, 4 * (len(resp_windows) - 1))

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
