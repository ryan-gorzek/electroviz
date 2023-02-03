
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import parse_SGLX_dir, read_Imec
from electroviz.utils.extractDigital import extractDigital
from electroviz.streams.digitalchannel import SyncChannel
import numpy as np

def imec_sets(
        paths, 
        save_path="", 
        sync_line=6, 
    ):
    """"""
    
    ap_onsets, ap_offsets = np.empty((0,)), np.empty((0,))
    lf_onsets, lf_offsets = np.empty((0,)), np.empty((0,))
    prev_ap, prev_lf = 0, 0
    for path in paths:
        
        _, imec_paths = parse_SGLX_dir(path)
        ap_metadata, ap_binary, lf_metadata, lf_binary, _ = read_Imec(imec_paths)
        ap_meta, ap_bnry, lf_meta, lf_bnry = ap_metadata[0], ap_binary[0], lf_metadata[0], lf_binary[0]

        num_samples = ap_bnry.shape[1]
        sampling_rate = float(ap_meta["imSampRate"])
        ap_sync_signal = extractDigital(ap_bnry, 
                                        0, num_samples-1, 
                                        0, 
                                        [sync_line], 
                                        ap_meta)
        ap_sync = SyncChannel(ap_sync_signal, sampling_rate, 0, 0.0)
        ap_ons = np.array(ap_sync.events["sample_onset"]) + prev_ap
        ap_offs = np.array(ap_sync.events["sample_offset"]) + prev_ap
        ap_onsets = np.concatenate((ap_onsets, ap_ons))
        ap_offsets = np.concatenate((ap_offsets, ap_offs))

        prev_ap += ap_sync_signal.size

        num_samples = lf_bnry.shape[1]
        sampling_rate = float(lf_meta["imSampRate"])
        lf_sync_signal = extractDigital(lf_bnry, 
                                        0, num_samples-1, 
                                        0, 
                                        [sync_line], 
                                        lf_meta)
        lf_sync = SyncChannel(lf_sync_signal, sampling_rate, 0, 0.0)
        lf_ons = np.array(lf_sync.events["sample_onset"]) + prev_lf
        lf_offs = np.array(lf_sync.events["sample_offset"]) + prev_lf
        lf_onsets = np.concatenate((lf_onsets, lf_ons))
        lf_offsets = np.concatenate((lf_offsets, lf_offs))

        prev_lf += lf_sync_signal.size

    np.savetxt(save_path + "ap_onsets.csv", ap_onsets)
    np.savetxt(save_path + "ap_offsets.csv", ap_offsets)
    np.savetxt(save_path + "lf_onsets.csv", lf_onsets)
    np.savetxt(save_path + "lf_offsets.csv", lf_offsets)

    