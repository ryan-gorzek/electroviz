
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
        stim_rand = stimulus.randomize()
        resp_rand_self = self_unit.get_response(stim_rand, time_window=(-100, 100), bin_size=1)
        resp_rand_other = other_unit.get_response(stim_rand, time_window=(-100, 100), bin_size=1)
    else:
        resp_rand_self = np.zeros(resp_self.shape)
        resp_rand_other = np.zeros(resp_other.shape)
    xcorr_rand = np.zeros((*self_responses.shape, rand_iters))
    for idx, (obs_self, obs_other, rand_self, rand_other) in enumerate(zip(resp_self, resp_other, resp_rand_self, resp_rand_other)):
        # Observed.
        obs_corr = correlate(obs_self, obs_other, mode="same", method="fft")
        obs_max = np.max(obs_corr)
        if np.max(xcorr_max) != 0:
            xcorr_obs[idx, :] = obs_corr / np.max(obs_corr)
        else:
            xcorr_obs[idx, :] = obs_corr
        # Random.
        if rand_iters > 0
            rand_corr = correlate(rand_self, rand_other, mode="same", method="fft")
            rand_max = np.max(rand_corr)
            if np.max(xcorr_max) != 0:
                xcorr_rand[idx, :] = rand_corr / np.max(rand_corr)
            else:
                xcorr_rand[idx, :] = rand_corr
    # Random iteration if requested.
    if rand_iters > 0:
        for i in range(rand_iters - 1):
            rand_idx = np.random.shuffle(range(resp_rand_self.shape[0]))
            resp_rand_self = resp_rand_self[rand_idx, :]
            resp_rand_other = resp_rand_other[rand_idx, :]
            for idx, (rand_self, rand_other) in enumerate(zip(resp_rand_self, resp_rand_other)):
                rand_corr = correlate(rand_self, rand_other, mode="same", method="fft")
                rand_max = np.max(rand_corr)
                if np.max(xcorr_max) != 0:
                    xcorr_rand[idx, :] = rand_corr / np.max(rand_corr)
                else:
                    xcorr_rand[idx, :] = rand_corr
        return np.nanmean(xcorr_obs, axis=0) - np.nanmean(xcorr_rand, axis=(0, 2))
    else:
        return np.nanmean(xcorr_obs, axis=0)

