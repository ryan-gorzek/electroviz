# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from scipy.stats import zscore


def plot_sparse_noise_response(
        unit, 
        stimulus, 
        time_window=[-0.050, 0.200], 
        bin_size=0.001, 
        colormap="inferno", 
    ):
    """"""

    sample_window = np.array(time_window)*30000
    num_samples = int(sample_window[1] - sample_window[0])
    num_bins = int(num_samples/(bin_size*30000))
    responses = np.zeros((num_bins, 14, 10, 2, 5))
    for event in stimulus:
        window = (sample_window + event.sample_onset).astype(int)
        resp = unit.get_spike_times(window)
        x_idx, y_idx, c_idx = stimulus.get_params_index((event.posx, event.posy, event.contrast))
        responses[:, x_idx, y_idx, c_idx, int(event.itrial)] = np.sum(resp.reshape(num_bins, -1), axis=1)
    mpl_use("Qt5Agg")
    fig, axs = plt.subplots(2, 1)
    kernels = np.zeros((14, 10, 2))
    for posx, posy, contrast in stimulus.unique:
        x_idx, y_idx, c_idx = stimulus.get_params_index((posx, posy, contrast))
        response_sum = np.nansum(responses[100:120, x_idx, y_idx, c_idx, :].squeeze(), axis=(0, 1))
        baseline_sum = np.nansum(responses[30:50, x_idx, y_idx, c_idx, :].squeeze(), axis=(0, 1))
        kernels[x_idx, y_idx, c_idx] += (response_sum - baseline_sum)
    axs[0].imshow(kernels[:, :, 0].T, cmap=colormap, clim=[kernels[:, :, 0].min(axis=(0, 1)), kernels[:, :, 0].max(axis=(0, 1))])
    axs[0].axis("off")
    axs[0].set_title("Off")
    axs[1].imshow(kernels[:, :, 1].T, cmap=colormap, clim=[kernels[:, :, 1].min(axis=(0, 1)), kernels[:, :, 1].max(axis=(0, 1))])
    axs[1].axis("off")
    axs[1].set_title("On")
    plt.show()

