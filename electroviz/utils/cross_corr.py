
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy.signal import correlate, correlation_lags, butter, lfilter
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

def cross_corr(
        units, 
        time_window=(-50, 50), 
        drop_stim=None, 
    ):
    """"""

    if drop_stim is not None:
        drop_onset = drop_stim[0].sample_onset - (500 * 30)
        drop_offset = drop_stim[-1].sample_onset + (500 * 30)
        drop_idx = np.arange(drop_onset, drop_offset, 1, dtype=int)
    else:
        drop_idx = []
    spikes_0 = units[0].get_spike_times(drop_idx=drop_idx).astype(int)
    spikes_1 = units[1].get_spike_times(drop_idx=drop_idx).astype(int)
    xcorr = correlate(spikes_0, spikes_1, mode="same", method="fft")
    lags = correlation_lags(spikes_0.size, spikes_1.size, mode="same")
    window = np.logical_and(lags >= time_window[0]*30, lags < time_window[1]*30)
    xcorr = xcorr[window].reshape(200, 15).sum(axis=1)
    b, a = butter(1, (75.0, 700), fs=2000, btype="bandpass")
    xcorr_filt = lfilter(b, a, xcorr)
    xcorr_filt[0:10] = 0
    xcorr_filt[191:201] = 0
    return xcorr, xcorr_filt

