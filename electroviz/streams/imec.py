
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_Imec, readMeta, makeMemMapRaw
from electroviz.utils.extractDigital import extractDigital
from electroviz.streams.digitalchannel import SyncChannel
import numpy as np
from electroviz.utils.gainCorrect import gainCorrectIM
from electroviz.streams.lfp import LFP

class ImecAP:
    """

    """


    digital_lines = dict({
                        "sync" : 6, 
                        })


    def __new__(
            self, 
            imec_path, 
        ):
        """"""

        # Create a list for storing objects derived from the probe.
        imec_ap = []
        # Read the Imec metadata and binary files.
        metadata, binary, _, _, _ = read_Imec(imec_path)
        for meta, bnry in zip(metadata, binary):
            num_samples = bnry.shape[1]
            sampling_rate = float(meta["imSampRate"])
            # Extract the sync channel first.
            sync_line = ImecAP.digital_lines["sync"]
            sync_signal = extractDigital(bnry, 
                                         0, num_samples-1, 
                                         0, 
                                         [sync_line], 
                                         meta)
            imec_ap.append(SyncChannel(sync_signal, sampling_rate))
        return imec_ap




class ImecLF:
    """

    """


    digital_lines = dict({
                        "sync" : 6, 
                        })


    def __new__(
            self, 
            imec_path, 
        ):
        """"""

        # Create a list for storing objects derived from the probe.
        imec_lf = []
        # Read the Imec metadata and binary files.
        _, _, metadata, binary, paths = read_Imec(imec_path)
        for meta, bnry, path in zip(metadata, binary, paths):
            num_samples = bnry.shape[1]
            sampling_rate = float(meta["imSampRate"])
            # Extract the sync channel first.
            sync_line = ImecLF.digital_lines["sync"]
            sync_signal = extractDigital(bnry, 
                                         0, num_samples-1, 
                                         0, 
                                         [sync_line], 
                                         meta)
            imec_lf.append(SyncChannel(sync_signal, sampling_rate))
            # Extract probe data.
            chan_list = np.arange(0, 384, 1)
            channels = gainCorrectIM(bnry, chan_list, meta, path + "/channels.mymemmap")
            channel_positions = np.load(path + "/channel_positions.npy")
            channel_map = np.load(path + "/channel_map.npy")
            imec_lf.append(LFP(channels[channel_map, :].squeeze(), channel_positions, sampling_rate))
        return imec_lf

