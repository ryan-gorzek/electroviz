
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
        bin_size=1, 
    ):
    """"""

    spikes_0 = units[0].bin_spikes(bin_size=bin_size, type="count")
    spikes_1 = units[1].bin_spikes(bin_size=bin_size, type="count")
    xcorr = correlate(spikes_0, spikes_1, mode="same", method="fft")
    lags = correlation_lags(spikes_0.size, spikes_1.size, mode="same")
    window = np.logical_and(lags >= time_window[0]/bin_size, lags <= time_window[1]/bin_size)
    b, a = butter(1, (75.0, 499.99), fs=1000/bin_size, btype="bandpass")
    xcorr_filt = lfilter(b, a, xcorr)
    return xcorr[window], xcorr_filt[window]

