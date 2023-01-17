# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import zscore
from scipy.signal import correlate, correlation_lags

def plot_corr_mat(
        pop_0, 
        pop_1, 
    ):
    """"""

    pop_0_spike_rates = zscore(pop_0._bin_spikes(), axis=1)
    pop_1_spike_rates = zscore(pop_1._bin_spikes(), axis=1)
    num_bins = pop_0_spike_rates.shape[1]
    corr_mat = np.matmul(pop_0_spike_rates, pop_1_spike_rates.T) / num_bins
    mpl_use("Qt5Agg")
    fig, ax = plt.subplots()
    ax.imshow(corr_mat, cmap="RdBu_r", clim=[-1, 1])
    ax.set_xlabel("Unit")
    ax.set_ylabel("Unit")
    cax = inset_axes(ax, width="5%", height="90%", loc="center right", borderpad=-5)
    colorbar = fig.colorbar(ax.images[0], cax=cax)
    colorbar.set_label("Pearson Correlation Coefficient", color="k", fontsize=14)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cax.tick_params(labelsize=14)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show(block=False)
    fig.set_size_inches(10, 8)

def plot_cross_corr(
        pop_0, 
        pop_1, 
        time_window=(-10, 10)
    ):
    """"""

    pop_0_spike_rates = pop_0._bin_spikes(bin_size=1)
    pop_1_spike_rates = pop_1._bin_spikes(bin_size=1)
    corr = correlate(pop_0_spike_rates[0, :], pop_1_spike_rates[0, :])
    lags = correlation_lags(pop_0_spike_rates.shape[1], pop_1_spike_rates.shape[1])
    plot_idx = (-10 <= lags) & (lags <= 10)
    corr = (corr / np.max(corr)) * np.corrcoef(pop_0_spike_rates[0, :], pop_1_spike_rates[0, :])[0, 1]
    mpl_use("Qt5Agg")
    fig, ax = plt.subplots()
    ax.plot(lags[plot_idx], corr[plot_idx])
    plt.show(block=False)
    fig.set_size_inches(8, 8)
