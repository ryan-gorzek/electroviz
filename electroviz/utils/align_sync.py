
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy.interpolate import interp1d
import copy

def align_sync(
        nidaq, 
        imec_sync, 
        nidaq_path, 
        type, 
    ):
    """"""

    if type == "AP":
        imec_onsets = np.loadtxt(nidaq_path + "/ap_onsets.csv")
        imec_offsets = np.loadtxt(nidaq_path + "/ap_offsets.csv")
    elif type == "LF":
        imec_onsets = np.loadtxt(nidaq_path + "/lf_onsets.csv")
        imec_offsets = np.loadtxt(nidaq_path + "/lf_offsets.csv")   

    prev_onsets, prev_offsets = 0, 0
    nidq = copy.deepcopy(nidaq)
    for ni in nidq:

        ni_onsets = np.array(ni[0].events["sample_onset"])
        ni_offsets = np.array(ni[0].events["sample_offset"])
        im_onsets = imec_onsets[prev_onsets : prev_onsets + ni_onsets.size]
        im_offsets = imec_offsets[prev_offsets : prev_offsets + ni_offsets.size]

        ni_correct = interp1d(ni_offsets, im_offsets, fill_value="extrapolate")
        
        for ni_signal in ni:
            ni_signal.events["sample_onset"] = ni_correct(np.array(ni_signal.events["sample_onset"]))
            ni_signal.events["sample_offset"] = ni_correct(np.array(ni_signal.events["sample_offset"]))

        prev_onsets += ni_onsets.size
        prev_offsets += ni_offsets.size

    return nidq
