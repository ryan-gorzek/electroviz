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
import warnings

class NIDAQ:
    """
    Interface for National Instruments DAQ (NI-DAQ) signals acquired via SpikeGLX.
    These data are stored in binary files (**.nidq.bin) and have accompanying metadata files (**.niqd.meta).
    SpikeGLX-derived files are processed herein by functions from the SpikeGLX Datafile Tools created by Bill Karsh.
    SpikeGLX is actively maintained and developed by Bill Karsh (https://github.com/billkarsh).
    """


    def __init__(
            self, 
            exp_path=os.getcwd(), 
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
        assert os.path.exists(exp_path), "Could not find the specified path to SpikeGLX NI-DAQ data."
        # Check for folder containing "ephys".
        ephys_name = None
        for subdir in os.listdir(exp_path):
            if "ephys" in subdir:
                ephys_subdir = subdir
        # Locate and read the meta file into a Python dictionary.
        meta_file = glob.glob(exp_path + ephys_subdir + "/*.nidq.meta")
        assert len(meta_file) == 1, "The **.nidq.meta file could not be read properly, check that it exists in the path without conflicts."
        self.meta_dict = SGLXReader.readMeta(meta_file[0])
        # Locate and read the binary file into a numpy memory map.
        bin_file = glob.glob(exp_path + ephys_subdir + "/*.nidq.bin")
        assert len(bin_file) == 1, "The **.nidq.bin file could not be read properly, check that it exists in the path without conflicts."
        self.bin_memmap = SGLXReader.makeMemMapRaw(bin_file[0], self.meta_dict)
        # Get some basic parameters from metadata for easy access.
        self.sampling_rate = float(self.meta_dict["niSampRate"])
        self.total_time = float(self.meta_dict["fileTimeSecs"])
        self.total_samples = int(self.sampling_rate * self.total_time)
        # Get parameters from each digital signal.
        self._get_digital_times(
            signal_names=["sync", "camera", "pc_clock", "photodiode"], 
            update_digital_signals=True, 
            )


    # def view():


    # get_channels():


    def plot_digital_channels(
            self,
            signal_names=["sync", "camera", "pc_clock", "photodiode"],
            time_window=[0, 100], 
        ):
        """

        """
        
        mpl_use("Qt5Agg")
        fig, axs = plt.subplots(len(signal_names), 1)
        for n, signal_name in enumerate(signal_names):
            signal, timepoints, _ = self._get_digital_signals(signal_name, time_window)
            axs[n].plot(timepoints, signal.squeeze())
            axs[n].set_title(signal_name)
            axs[n].set_xlabel("Time (s)")
            axs[n].set_ylim((-0.1, 1.1))
            axs[n].set_yticks((0, 1))
            axs[n].set_ylabel("Signal Value")
        fig.set_size_inches(12, 6)
        plt.show()



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
            X = np.linspace(0, self.total_time, counts_full.size)
            mpl_use("Qt5Agg")
            fig, ax = plt.subplots()
            ax.plot(X, counts_full)
            ax.set_title("Sampling Rate Stability over Recording")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("# of Samples in Sync Pulse")
            #### Add probability-normalized histogram?
            fig.set_size_inches(10, 4)
            plt.show()
        return counts_full #### Make this into a df with numbered index and 0/1 indicator column

    def _get_digital_signals(
            self, 
            signal_names=["sync", "camera", "pc_clock", "photodiode"],
            time_window=[0, None]
        ):

        # Get sample index from time window.
        sample_window = self._get_time_sample(time_window, return_range=False)
        # Get specified digital lines across all timepoints.
        line_num = self._parse_signal_IDs(signal_names)
        signals = SGLXReader.ExtractDigital(SGLXReader(), 
                                            self.bin_memmap, 
                                            sample_window[0], sample_window[1], 
                                            0, 
                                            line_num, 
                                            self.meta_dict)
        # Get sample indices.
        sample_idx = np.arange(sample_window[0], sample_window[1]+1, 1)
        # Get timepoints.
        timepoints = self._get_sample_time(sample_idx)
        return signals, timepoints, sample_idx


    def _get_digital_times(
            self,
            signal_names=["sync", "camera", "pc_clock", "photodiode"],
            blank=False, 
            update_digital_signals=True,
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

        # Create a dictionary to store parameters for each signal.
        signals_dict = {}
        # Iterate through digital signals and extract onsets and offsets.
        for signal_name, signal in zip(signal_names, signals):
            # Get value for digital signal at recording start and end.
            value_start = signal[0]
            value_end = signal[-1]
            # Sync runs continuously and can start or end at 1, CHECK ON CAMERA???
            # Raise if this happens for the pc_clock or photodiode because it could indicate an issue.
            value_error = "The {0} signal unexpectedly has a {1} value of {2:b} in this recording, " + \
                              "inspect the data in SpikeGLX for potential problems."
            if (signal_name != "sync") & (signal_name != "camera") & (value_start != 0):
                raise Exception(value_error.format(signal_name, "starting", value_start))
            if (signal_name != "sync") & (signal_name != "camera") & (value_end != 0):
                raise Exception(value_error.format(signal_name, "ending", value_end))
            
            # Immediately check for blank.
            #     sync should never have a blank, while camera should always have a blank.
            if (blank == True) | (signal_name == "camera"):
                # Take temporal difference of digital signal for locating onsets and offsets.
                signal_diff = np.insert(np.diff(-1 * signal), 0, 0)
                value_start, value_end = value_end, value_start
                (sample_onsets,) = np.where(signal_diff == 1)
                (sample_offsets,) = np.where(signal_diff == -1)
                sample_offsets += -1
                #### What about camera starting or ending on 1???
                #### Needs work
            elif (blank == False) | (signal_name == "sync"):
                # Take temporal difference of digital signal for locating onsets and offsets.
                signal_diff = np.insert(np.diff(signal), 0, 0)
                # A stimulus without a blank should start 0->1, then flip up-and-down, 
                #     making any non-zero value of the diff an onset, except the last.
                (sample_onsets_all,) = np.where(signal_diff != 0)
                # The last "onset" (non-zero value) is really the first sample after the final stimulus, 
                #     so we drop it.
                sample_onsets = sample_onsets_all[:-1]
                # Without a blank, the offsets always immediately precede the onsets by one sample, 
                #     except the first onset, which is the first stimulus.
                sample_offsets = sample_onsets_all[1:] - 1
                # The sync signal is different. It runs continuously and the recording might start
                #     while it is 0 or 1, so we also include the onsets and offsets that we excluded 
                #     for the triggered stimuli that always begin with 0->1.
                if signal_name == "sync":
                    sample_onsets = sample_onsets_all[:-1]
                    sample_offsets = sample_onsets_all[1:] - 1
            else:
                raise Exception("Onsets and offsets could not be extracted for {0}, " + 
                                    "check this signal and parameters for errors.".format(signal_name))
            
            # Get the number of samples in each pulse.
            sample_duration = (sample_offsets - sample_onsets) + 1
            # Get the onset and offset times (in seconds) relative to the start of the recording.
            time_onsets = self._get_sample_time(sample_onsets)
            time_offsets = self._get_sample_time(sample_offsets)
            # Get the duration of each pulse in seconds.
            time_duration = time_offsets - time_onsets
            # Get the value of each pulse by taking the mean of each range (onset, offset).
            value_array = np.zeros((sample_onsets.size,))
            for idx, (onset, offset) in enumerate(zip(sample_onsets, sample_offsets)):
                value = np.mean(signal[onset:offset+1])
                if (value == 0) | (value == 1):
                    value_array[idx] = int(value)
                else:
                    warnings.warn("Mean digital signal during a pulse is expected to be 0 or 1 but has a value of {0:f}.\n".format(value) + 
                                  "    Assigning anyway, but this may indicate a problem with NI-DAQ digital data streams.")
                    print(signal_name, signal, onset, offset, value)
                    value_array[idx] = value

            # Add signal parameters to a dictionary:
            params_dict = {
                "sample_onsets" :    sample_onsets.astype(int),    # (# of samples to onset from sample 0, the start of recording)
                "sample_offsets" :   sample_offsets.astype(int),   # (# of samples to onset from sample 0, the start of recording)
                "sample_durations" : sample_duration.astype(int),  # (# of samples from onset to offset, inclusive)
                "time_onsets" :      time_onsets,       # (seconds to sample onset, from time 0, the start of recording)
                "time_offsets" :     time_offsets,      # (seconds to sample onset, from time 0, the start of recording)
                "time_durations" :   time_duration,     # (seconds from onset to offset, ???)
                "digital_value" :    value_array,       # (0 or 1, should alternate when there is no blank)
                            }
            # Append signals dictionary to be returned by this function.
            signals_dict[signal_name] = params_dict
            # Update class attribute dictionary if specified.
            if update_digital_signals == True:
                self.digital_signals[signal_name].update(params_dict)
        return signals_dict


    def _get_sample_time(
            self, 
            sample_num, 
        ):
        """

        """

        sample_length = 1/self.sampling_rate
        sample_times_all = np.arange(0, self.total_time, sample_length, dtype=float)
        if self.total_samples != sample_times_all.size:
            warnings.warn("Sample times array does not match the total number of samples.")
        sample_times = sample_times_all[sample_num]
        return sample_times

    def _get_time_sample(
            self, 
            time_window=(0, None), 
            return_range=True, 
        ):
        """

        """

        sample_length = 1/self.sampling_rate
        sample_times_all = np.arange(0, self.total_time, sample_length, dtype=float)
        if self.total_samples != sample_times_all.size:
            warnings.warn("Sample times array does not match the total number of samples.")
        
        def find_min_dist_1D(number, array):
            dists = np.abs(array - number)
            (min_idx,) = np.where(dists == np.min(dists))
            return int(min_idx)
        
        # Sample onset.
        if time_window[0] is None:
            sample_onset = 0
        else:
            sample_onset = find_min_dist_1D(time_window[0], sample_times_all)
        # Sample offset.
        if time_window[1] is None:
            sample_offset = sample_times_all.size
        else:
            sample_offset = find_min_dist_1D(time_window[1], sample_times_all)
        # Return range if specified, otherwise return tuple matching time_window input.
        if range == True:
            time_samples = np.arange(sample_onset, sample_offset, 1)
        else:
            time_samples = (sample_onset, sample_offset)


        
        return time_samples


    def _parse_signal_IDs(
            self, 
            signal_IDs, 
        ):
        """
        If it's a single ID, not a list, make it one so other methods can expect list.
        If signal name(s) is specified swap it for the line_number(s). Otherwise pass.
        """

        if not isinstance(signal_IDs, list):
            signal_IDs = [signal_IDs]
        line_numbers = [self.digital_signals[ID]["line_num"] 
                       if isinstance(ID, str) else ID
                       for ID in signal_IDs]
        return line_numbers
