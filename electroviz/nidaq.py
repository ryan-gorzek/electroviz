# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
import glob
from .utils import SGLXReader
from skimage.measure import regionprops

class NIDAQ:
    """
    Interface for National Instruments DAQ (NI-DAQ) signals acquired via SpikeGLX.
    These data are stored in binary files (**.nidq.bin) and have accompanying metadata files (**.niqd.meta).
    SpikeGLX-derived files are processed herein by functions from the SpikeGLX Datafile Tools created by Bill Karsh.
    SpikeGLX is actively maintained and developed by Bill Karsh (https://github.com/billkarsh).
    """

    def __init__(
            self, 
            nidq_path=os.getcwd(), 
        ):
        """
        Constructor reads NI-DAQ binary and metadata files (both must be present) from SpikeGLX.
            **.nidq.meta is read into a Python dictionary.
            **.nidq.bin is read into a numpy memory-map.
        """

        self.digital_signals = dict({
            "sync" :       {"line_num":7}, 
            "camera" :     {"line_num":5}, 
            "pc_clock" :   {"line_num":4}, 
            "photodiode" : {"line_num":1}, 
                                 })

        # Load SpikeGLX binary and metadata files.
        assert os.path.exists(nidq_path), "Could not find the specified path to SpikeGLX NI-DAQ data."
        # Locate and read the meta file into a Python dictionary.
        meta_file = glob.glob(nidq_path + "/*.nidq.meta")
        assert len(meta_file) == 1, "The **.nidq.meta file could not be read properly, check that it exists in the path without conflicts."
        self.meta_dict = SGLXReader.readMeta(meta_file[0])
        # Locate and read the binary file into a numpy memory map.
        bin_file = glob.glob(nidq_path + "/*.nidq.bin")
        assert len(bin_file) == 1, "The **.nidq.bin file could not be read properly, check that it exists in the path without conflicts."
        self.bin_memmap = SGLXReader.makeMemMapRaw(bin_file[0], self.meta_dict)
        # Get some basic parameters from metadata for easy access.
        self.sampling_rate = float(self.meta_dict["niSampRate"])
        self.recording_len = float(self.meta_dict["fileTimeSecs"])
        self.total_samples = int(self.sampling_rate * self.recording_len)

    def check_sampling_rate(
            self, 
        ):
        """
        Check the sampling rate over time using the sync signal.
        This signal should be a 1 Hz pulse. 
        If the length of the pulse varies, the sampling rate is inconsistent across the recording.
        """
        sync_idx = self.digital_signals["sync"]["line_num"]
        signal = SGLXReader.ExtractDigital(SGLXReader(), 
                                           self.bin_memmap, 
                                           0, self.total_samples-1, 
                                           0, 
                                           [sync_idx], 
                                           self.meta_dict)
        regions = regionprops(signal)
        return regions, signal

    # def get_times(
    #         self,
    #         signals=self.digital_signals.keys(),
    #         sets=["on", "off"], 
    #         blank=False, 
    #     ):
    #     """"""
    #     for sig in signals:
            
    # get_channels

    # plot_channels
