# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
import glob
from .utils import SGLXReader
import numpy as np
from skimage.morphology import label as sk_label
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from warnings import warn

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


    def check_time_stability(
            self, 
            signal_name="sync", 
            blank=False, 
            plot=False, 
        ):
        """
        Check any digital signal's stability over time by comparing the number of samples encompassed by each pulse.

        This is particularly useful for checking the sampling rate over time using the sync signal (default).
            This signal should be a 1 Hz pulse with a 0.5 s duration.
            If the length of the pulse varies, the sampling rate is inconsistent across the recording.

        This is probably not useful for stimuli whose duration varies randomly. 
        """

        # Get the sync digital signal.
        line_num = self._parse_signal_IDs(signal_name)
        signal = SGLXReader.ExtractDigital(SGLXReader(), 
                                           self.bin_memmap, 
                                           0, self.total_samples-1, 
                                           0, 
                                           line_num, 
                                           self.meta_dict)
        # First, there is never a blank in the sync signal.
        assert (line_num == 7) != (blank == True), "There is never a blank in the sync signal, did you mean to specify blank=False?"
        # If there is no blank, both 0 and 1 indicate a stimulus.
        # In this case, interleave sample counts depending on starting point (0 or 1).
        if blank == False:
            labels = sk_label(signal[0])
            labels_a = np.abs(labels - signal[0, 0])
            _, counts_a = np.unique(labels - signal[0, 0], return_counts=True)
            labels_b = np.abs(labels - signal[0, 0])
            _, counts_b = np.unique(labels - np.abs(signal[0, 0]-1), return_counts=True)
            counts_full = np.zeros(counts_a.size + counts_b.size - 2, dtype=int)
            # Counting 1's in both cases, so leave out the 0 (first) count either way.
            counts_full[0::2] += counts_a[1:]
            counts_full[1::2] += counts_b[1:]
            # For these quality control purposes, we'll set the first and last counts to NaN.
            #     - the sync signal is almost always cut off at both ends, which is expected.
            #     - the first stimulus always (???) starts with a pulse to 1
            #     - the last stimulus always ends going back to zero
            # These sources of variation with squeeze the y-axis and preclude close examination.
            counts_full[(1, -1)] = (np.nan, np.nan)
        # If there is a blank, 1's indicate a stimulus, so just count these pulses.
        else:
            labels = sk_label(signal[0])
            _, counts = np.unique(labels, return_counts=True)
            counts_full = counts[1:]

        # If specified, make a line plot to show the sampling rate stability over the recording.
        if plot == True:
            #### Need to align X with center of pulses (not regionprops, maybe manual)
            X = np.linspace(0, self.recording_len, counts_full.size)
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
            ax.plot(X, counts_full)
            ax.set_title("Sampling Rate Stability over Recording")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("# of Samples in Sync Pulse")
            fig.set_size_inches(10, 4)
            plt.show()
        return counts_full #### Make this into a df with numbered index and 0/1 indicator column

    def get_digital_times(
            self,
            signal_names=["sync", "camera", "pc_clock", "photodiode"],
            blank=False, 
        ):
        """
        
        """

        # Get specified digital lines across all timepoints.
        line_num = self._parse_signal_IDs(signal_names)
        signals = SGLXReader.ExtractDigital(SGLXReader(), 
                                            self.bin_memmap, 
                                            0, self.total_samples-1, 
                                            0, 
                                            line_num, 
                                            self.meta_dict)
        # Iterate through digital signals and extract onsets and offsets.
        for signal_name, signal in zip(signal_names, signals):
            # Get value for digital signal at recording start and end.
            value_start = signal[0]
            value_end = signal[-1]
            # Sync can start or end at 1, but raise if this happens for camera, pc_clock, or photodiode.
            value_error = "The {0} signal unexpectedly has a starting value of {1:b} in this recording, " + 
                              "inspect the data in SpikeGLX for potential problems."
            if (signal_name != "sync") & (value_start != 0):
                 raise Exception(value_warning.format(signal_name, value_start))
            if (signal_name != "sync") & (value_end != 0):
                 raise Exception(value_warning.format(signal_name, value_start))
            
            # Take temporal difference of digital signal for locating onsets and offsets.
            signal_diff = np.insert(np.diff(signal), 0, 0)
            # Immediately check for blank.
            #     sync should never have a blank, while camera should always have a blank.
            if (blank == True) | (signal == "camera"):
                 (sample_onsets,) = np.where(signal_diff == 1)
                 (sample_offsets,) = np.where(signal_diff == -1) -= 1
            elif (blank == False) | (signal != "sync"):
                 # A stimulus without a blank should start 0->1

            elif signal == "sync"
                # Check whether sync starts on 0 or 1

                # The first and last sample of sync will always be an onset and offset, respectively
                sample_onsets = np.insert(sample_onsets, 0, 0)
                sample_offsets = np.append(sample_offsets, signal.size - 1)
            else:
                raise Exception("Onsets and offsets could not be extracted for {s}, " + 
                                    "check this signal and parameters for errors.".format(s=sig))

        # Create dataframe for each digital signal, storing:
        #     index (implicit)
        #     samples to onset    (#, from sample 0, the start of recording)
        #     samples to offset   (#, from sample 0, the start of recording)
        #     sample duration     (#, from onset to offset, inclusive)
        #     time (s) to onset   (s, from time 0, the start of recording)
        #     time (s) to offset  (s, from time 0, the start of recording)
        #     time duration       (s, from onset to offset, inclusive)
        #     value               (0 or 1, should alternate when there is no blank)
        
            
    # get_channels

    # plot_channels

    def _get_sample_time(
            self, 
            sample_num, 
        ):
        """

        """
        

    def _parse_signal_IDs(
            self, 
            signal_IDs, 
        ):
        """
        If it's a single ID, not a list, make it one so other methods can expect list.
        If signal name(s) is specified swap it for the line_number(s). Otherwise pass.
        """

        if (len(signal_IDs) == 1) & (not isinstance(signal_IDs, list)):
            signal_IDs = [signal_IDs]

        line_numbers = [self.digital_signals[ID]["line_num"] 
                       if isinstance(ID, str) else ID
                       for ID in signal_IDs]
        return line_numbers


