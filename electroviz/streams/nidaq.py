# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_NIDAQ, readMeta, makeMemMapRaw
from electroviz.utils.extractDigital import extractDigital
from electroviz.streams.digitalchannel import DigitalChannel, SyncChannel

class NIDAQ:
    """

    """

    digital_lines = dict({
                        "sync" : 7, 
                        # "camera" : 5, 
                        "pc_clock" : 4, 
                        "photodiode" : 1, 
                        })

    
    def __new__(
            self, 
            nidaq_path, 
        ):
        """"""
        
        # Read the NIDAQ metadata and binary files.
        metadata, binary, offsets = read_NIDAQ(nidaq_path)
        num_samples = binary.shape[1]
        sampling_rate = float(metadata["niSampRate"])
        # Create a list for storing objects derived from the NIDAQ.
        nidaq = []
        # Extract the sync channel first.
        sync_line = NIDAQ.digital_lines["sync"]
        sync_signal = extractDigital(binary, 
                                     0, num_samples-1, 
                                     0, 
                                     [sync_line], 
                                     metadata)
        nidaq.append(SyncChannel(sync_signal, sampling_rate))
        # Get concatenation times if applicable.
        if offsets is not None:
            offsets = offsets[1][1:].astype(float)
        # Extract other specified digital channels.
        digital_lines = [NIDAQ.digital_lines[name] for name in NIDAQ.digital_lines.keys() if name != "sync"]
        for line in digital_lines:
            digital_signal = extractDigital(binary, 
                                            0, num_samples-1, 
                                            0, 
                                            [line], 
                                            metadata)
            nidaq.append(DigitalChannel(digital_signal, sampling_rate, concat_times=offsets))
        return nidaq
