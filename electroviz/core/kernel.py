# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from scipy.stats import zscore


class Kernel:
    """

    """


    def __init__(
            self, 
            unit, 
            stimulus, 
            time_window=[-0.050, 0.200], 
            bin_size=0.001, 
        ):
        """"""

        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        self.responses = np.zeros((num_bins, *stimulus.shape))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = unit.get_spike_times(window)
            self.responses[:, *event.stim_indices] = np.sum(resp.reshape(num_bins, -1), axis=1)

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
            bin_size=0.001, 
            resp_window=[0.050, 0.070], 
            base_window=[-0.030, 0], 
        ):
        """"""
        
        super().__init__(
            unit, 
            stimulus, 
            time_window=time_window, 
            bin_size=bin_size, 
                       )

        kernels = np.zeros(stimulus.shape[0:3])
        sample_window = np.array(time_window)*30000
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size*30000))
        resp_on, resp_off = self._time_to_bins(resp_window, time_window, num_bins)
        base_on, base_off = self._time_to_bins(base_window, time_window, num_bins)
        for stim_indices in np.ndindex(stimulus.shape[:3]):
            resp_count = self.responses[resp_on:resp_off, *stim_indices, :].sum(axis=0).mean()
            base_count = self.responses[base_on:base_off, *stim_indices, :].sum(axis=0).mean()
            kernels[*stim_indices] += (resp_count - base_count)
        self.OFF = kernels[0]
        self.ON = kernels[1]

    # def _fit_2D_gaussian(
    #         self, 
    #     ):
    #     """"""

    def plot_raw(
            self, 
            cmap="inferno", 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(self.OFF.T, cmap=cmap, clim=[self.OFF.min(axis=(0, 1)), self.OFF.max(axis=(0, 1))])
        axs[0].axis("off")
        axs[0].set_title("Off")
        axs[1].imshow(self.ON.T, cmap=cmap, clim=[self.ON.min(axis=(0, 1)), self.ON.max(axis=(0, 1))])
        axs[1].axis("off")
        axs[1].set_title("On")
        plt.show(block=False)
        # fig.set_size_inches()
        # if 



