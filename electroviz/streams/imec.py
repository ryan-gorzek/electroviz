# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_Imec, readMeta, makeMemMapRaw
from electroviz.utils.extractDigital import extractDigital
from electroviz.streams.digitalchannel import SyncChannel

class Imec:
    """

    """

    digital_lines = dict({
                        "sync" : 7, 
                        })

    
    def __new__(
            self, 
            imec_path, 
        ):

        # Read the Imec metadata and binary files.
        metadata, binary = read_Imec(imec_path)
        num_samples = binary.shape[1]
        sampling_rate = float(metadata["imSampRate"])
        # Create a list for storing objects derived from the probe.
        imec = []
        # Extract the sync channel first.
        sync_line = Imec.digital_lines["sync"]
        sync_signal = extractDigital(binary, 
                                     0, num_samples-1, 
                                     0, 
                                     [sync_line], 
                                     metadata)
        imec.append(SyncChannel(sync_signal, sampling_rate))
        return imec
