
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_NIDAQ, readMeta, makeMemMapRaw
from electroviz.utils.extractDigital import extractDigital
from electroviz.streams.digitalchannel import DigitalChannel, SyncChannel
import numpy as np

class NIDAQ:
    """

    """


    def __new__(
            self, 
            nidaq_path, 
            nidaq_gates, 
        ):
        """"""

        self.digital_lines = dict({
                        "sync" : 7, 
                        "pc_clock" : 4, 
                        "photodiode" : 1, 
                        "led" : 6, 
                                   })
        
        # Read the NIDAQ metadata and binary files.
        binarys, metadatas = read_NIDAQ(nidaq_path, nidaq_gates)
        nidaq = []
        sample_start, time_start = 0, 0.0
        for binary, metadata in zip(binarys, metadatas):
            num_samples = binary.shape[1]
            sampling_rate = float(metadata["niSampRate"])
            # Create a list for storing objects derived from the NIDAQ.
            gate = []
            # Extract the sync channel first.
            sync_line = self.digital_lines["sync"]
            sync_signal = extractDigital(binary, 
                                        0, num_samples-1, 
                                        0, 
                                        [sync_line], 
                                        metadata)
            gate.append(SyncChannel(sync_signal, sampling_rate, sample_start, time_start))
            # Extract other specified digital channels.
            digital_lines = [self.digital_lines[name] for name in self.digital_lines.keys() if name != "sync"]
            for line in digital_lines:
                digital_signal = extractDigital(binary, 
                                                0, num_samples-1, 
                                                0, 
                                                [line], 
                                                metadata)
                if np.any(digital_signal != 0):
                    channel = DigitalChannel(digital_signal, sampling_rate, sample_start, time_start)
                    if np.any(channel.events["sample_duration"] == 1):
                        event_idx = np.where(channel.events["sample_duration"] == 1)[0]
                        for ev_idx in event_idx:
                            sample_idx = channel.events["sample_onset"][ev_idx]
                            digital_signal = channel.signal
                            digital_signal[sample_idx - 4] = 1 - channel.events["digital_value"][ev_idx]
                        channel = DigitalChannel(digital_signal, sampling_rate, sample_start, time_start)
                    gate.append(channel)
            nidaq.append(gate)
            sample_start += len(digital_signal)
            time_start += len(digital_signal) / sampling_rate
        return nidaq

