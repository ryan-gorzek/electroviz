
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy.signal import correlate, correlation_lags, butter, lfilter

def cross_corr(
        units, 
        time_window=(-50, 50), 
        bin_size=0.5, 
    ):
    """"""

    spikes_0 = units[0].bin_spikes(bin_size=bin_size, type="count")
    spikes_1 = units[1].bin_spikes(bin_size=bin_size, type="count")
    xcorr = correlate(spikes_0, spikes_1, mode="same", method="fft")
    lags = correlation_lags(spikes_0.size, spikes_1.size, mode="same")
    window = np.logical_and(lags >= time_window[0]/bin_size, lags <= time_window[1]/bin_size)
    b, a = butter(1, (75.0, 750), fs=1000/bin_size, btype="bandpass")
    xcorr_filt = lfilter(b, a, xcorr)
    return xcorr[window], xcorr_filt[window]


def cross_corr_stim(
        self_unit, 
        other_unit, 
        stimulus, 
        time_window=(-100, 100), 
        bin_size=1, 
        rand_iters=0, 
        return_raw=False, 
    ):
    """"""

    responses = {}
    responses["self_obs"] = self_unit.get_response(stimulus, time_window=time_window, bin_size=bin_size)
    responses["other_obs"] = other_unit.get_response(stimulus, time_window=time_window, bin_size=bin_size)
    xcorr_obs, xcorr_rand = np.zeros((responses["self_obs"].shape)), np.zeros((*responses["self_obs"].shape, 1))
    if rand_iters > 0:
        responses["self_rand"] = self_unit.get_response(stimulus.randomize(), time_window=time_window, bin_size=bin_size)
        responses["other_rand"] = other_unit.get_response(stimulus.randomize(), time_window=time_window, bin_size=bin_size)
        xcorr_rand = np.zeros((*responses["self_obs"].shape, rand_iters))
    for idx, (self_obs, other_obs) in enumerate(zip(responses["self_obs"], responses["other_obs"])):
        # Observed.
        obs_corr = correlate(self_obs, other_obs, mode="same", method="fft")
        xcorr_obs[idx, :] = obs_corr
        # Random.
        if rand_iters > 0:
            self_rand = responses["self_rand"][idx, :]
            other_rand = responses["other_rand"][idx, :]
            rand_corr = correlate(self_rand, other_rand, mode="same", method="fft")
            xcorr_rand[idx, :, 0] = rand_corr
    # Random iteration if requested.
    if rand_iters > 1:
        for i in range(1, rand_iters):
            self_rand_idx = list(range(responses["self_rand"].shape[0]))
            other_rand_idx = list(range(responses["self_rand"].shape[0]))
            np.random.shuffle(self_rand_idx), np.random.shuffle(other_rand_idx)
            for idx, (self_rand, other_rand) in enumerate(zip(responses["self_rand"][self_rand_idx, :], 
                                                              responses["other_rand"][other_rand_idx, :])):
                rand_corr = correlate(self_rand, other_rand, mode="same", method="fft")
                xcorr_rand[idx, :, i] = rand_corr
    if return_raw is False:
        return xcorr_obs.mean(axis=0) - xcorr_rand.mean(axis=(0, 2))
    else:
        return xcorr_obs.mean(axis=0) - xcorr_rand.mean(axis=(0, 2)), xcorr_obs.mean(axis=0)

