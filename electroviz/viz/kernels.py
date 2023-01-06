# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from scipy.stats import zscore

def plot_population_response(
        population, 
        stimulus, 
        time_window=[-0.050, 0.200], 
        bin_size=0.1, 
    ):
    """"""

    sample_window = np.array(time_window)*30000
    num_samples = int(sample_window[1] - sample_window[0])
    num_bins = int(num_samples/(bin_size*30000))
    responses = np.zeros((num_bins, len(stimulus)))
    for event in stimulus:
        window = (sample_window + event.sample_onset).astype(int)
        resp = population.spike_times[:, window[0]:window[1]].sum(axis=0).squeeze()
        responses[:, event.index] = np.sum(resp.reshape(num_bins, -1), axis=1).squeeze()

    mpl_use("Qt5Agg")
    fig, axs = plt.subplots()
    mean_response = np.nanmean(responses.squeeze(), axis=1)
    # axs.axvline(50, color="k", linestyle="dashed")
    axs.plot(mean_response, color="b")
    axs.set_xlabel("Time from onset (ms)")
    axs.set_xticks([0, 50, 100, 150, 200, 250])
    axs.set_xticklabels([-50, 0, 50, 100, 150, 200])
    plt.show()

def plot_population_raster(
        population, 
        stimulus, 
        time_window=[-0.050, 0.200], 
        bin_size=0.1, 
    ):
    """"""

    sample_window = np.array(time_window)*30000
    num_samples = int(sample_window[1] - sample_window[0])
    num_bins = int(num_samples/(bin_size*30000))
    responses = np.zeros((num_bins, len(population), len(stimulus)))
    for event in stimulus:
        window = (sample_window + event.sample_onset).astype(int)
        resp = population.spike_times[:, window[0]:window[1]]
        for unit_idx, unit_resp in enumerate(resp):
            responses[:, unit_idx, event.index] = np.sum(unit_resp.reshape(num_bins, -1), axis=1).squeeze()

    mpl_use("Qt5Agg")
    fig, axs = plt.subplots()
    mean_response = np.nanmean(responses, axis=2)
    z_response = zscore(mean_response.T, axis=1)
    axs.imshow(z_response, cmap="hot")
    plt.show()

def plot_sparse_noise_response(
        unit, 
        stimulus, 
        time_window=[-0.050, 0.200], 
    ):
    """"""

    sample_window = np.array(time_window)*30000
    num_samples = int(sample_window[1] - sample_window[0])
    responses = np.zeros((int(num_samples/30), 14, 10, 2, 5))
    for event in stimulus:
        window = (sample_window + event.sample_onset).astype(int)
        resp = unit.get_spike_times(window)
        x_idx, y_idx, c_idx = stimulus.get_params_index((event.posx, event.posy, event.contrast))
        responses[:, x_idx, y_idx, c_idx, int(event.itrial)] = np.sum(resp.reshape(250, -1), axis=1)
    
    mpl_use("Qt5Agg")
    fig, axs = plt.subplots(2, 1)
    kernels = np.zeros((14, 10, 2))
    for posx, posy, contrast in stimulus.unique:
        x_idx, y_idx, c_idx = stimulus.get_params_index((posx, posy, contrast))
        response_sum = np.nansum(responses[100:150, x_idx, y_idx, c_idx, :].squeeze(), axis=(0, 1))
        baseline_sum = np.nansum(responses[0:50, x_idx, y_idx, c_idx, :].squeeze(), axis=(0, 1))
        kernels[x_idx, y_idx, c_idx] += (response_sum - baseline_sum)
    axs[0].imshow(kernels[:, :, 0].T, cmap="hot")
    axs[0].axis("off")
    axs[1].imshow(kernels[:, :, 1].T, cmap="hot")
    axs[1].axis("off")
    plt.show()

