
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
    ):
    """"""

    self_responses = self_unit.get_response(stimulus, time_window=(-100, 100), bin_size=1)
    other_responses = other_unit.get_response(stimulus, time_window=(-100, 100), bin_size=1)
    xcorr_obs = np.zeros((self_responses.shape))
    for idx, (self_resp, other_resp) in enumerate(zip(self_responses, other_responses)):
        xcorr = correlate(self_resp, other_resp, mode="same", method="fft")
        print(xcorr)
        xcorr_obs[idx, :] = xcorr / np.max(xcorr)
    return xcorr_obs

