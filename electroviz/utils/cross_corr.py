
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy.signal import correlate

def cross_corr(
        self_unit, 
        other_unit, 
        stimulus, 
        time_window=(-100, 100), 
        bin_size=1, 
        rand_iters=0, 
    ):
    """"""

    # Observed.
    resp_self = self_unit.get_response(stimulus, time_window=(-100, 100), bin_size=1)
    resp_other = other_unit.get_response(stimulus, time_window=(-100, 100), bin_size=1)
    xcorr_obs = np.zeros((self_responses.shape))
    # Random
    if rand_iters > 0:
        rand_stim = stimulus.randomize()
        resp_rand_self = self_unit.get_response(rand_stim, time_window=(-100, 100), bin_size=1)
        resp_rand_other = other_unit.get_response(rand_stim, time_window=(-100, 100), bin_size=1)
        xcorr_rand = np.zeros((*self_responses.shape, rand_iters))
    else:
        resp_rand_self = None
    for idx, (obs_self, obs_other, rand_self, rand_other) in enumerate(zip(resp_self, resp_other, resp_rand_self, resp_rand_other)):
        # Observed.
        obs_corr = correlate(obs_self, obs_other, mode="same", method="fft")
        obs_max = np.max(obs_corr)
        if np.max(xcorr_max) != 0:
            xcorr_obs[idx, :] = obs_corr / np.max(obs_corr)
        else:
            xcorr_obs[idx, :] = obs_corr
        if rand_iters > 0
        # Random.
        rand_corr = correlate(rand_self, rand_other, mode="same", method="fft")
        rand_max = np.max(rand_corr)
        if np.max(xcorr_max) != 0:
            xcorr_rand[idx, :] = rand_corr / np.max(rand_corr)
        else:
            xcorr_rand[idx, :] = rand_corr
    # Random iteration if requested.
    if rand_iters > 0:
        for i in range(rand_iters - 1):

    return np.nanmean(xcorr_obs) - np.nanmean(xcorr_rand)

