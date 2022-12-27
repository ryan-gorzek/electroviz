# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from electroviz.core.event import Event
from skimage.morphology import label as sk_label
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
import warnings
import pandas as pd

class DigitalChannel:
    """

    """


    def __init__(
            self, 
            signal, 
            sampling_rate, 
        ):
        """
    
        """

        # Get some basic data and parameters for easy access.
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.total_samples = len(signal)
        self.total_time = self.total_samples / self.sampling_rate
        self._build_events_df()
        self._get_events()

    
    def __getitem__(
            self, 
            index, 
        ):
        """"""
        return self._Events[index]

    
    def __iter__(
            self, 
        ):
        """"""
        self._Events_num = 0
        return iter(self._Events)


    def __next__(
            self, 
        ):
        """"""
        
        if self._Events_num > len(self._Events):
            raise StopIteration
        else:
            event = self.Events[self._Events_num]
            self._Events_num += 1
        return event


    def drop_and_rebuild(
            self, 
            drop_samples, 
        ):
        """
        Aligning NIDAQ and Imec synchronization channels requires dropping some samples. This method accepts 
        sample indices to drop and rebuilds attributes accordingly.
        """

        # Remove specified samples from the digital signal.
        self.signal = np.delete(self.signal, drop_samples)
        # Update signal parameters.
        self.total_samples = len(self.signal)
        self.total_time = self.total_samples / self.sampling_rate
        # Update the event times dataframe.
        self.events = self._get_digital_events()


    def _build_events_df(
            self, 
        ):
        """
        
        """

        # Get value for digital signal at recording start and end.
        value_start = self.signal[0]
        value_end = self.signal[-1]
        # Take temporal difference of digital signal for locating onsets and offsets.
        signal_diff = np.insert(np.diff(self.signal), 0, 0)
        # A stimulus should start 0->1, then flip up-and-down, making any non-zero value of the diff an onset, 
        #     except the last.
        (sample_onsets_all,) = np.nonzero(signal_diff != 0)
        # The last "onset" (non-zero value) is really the first sample after the final stimulus, so we drop it.
        sample_onsets = sample_onsets_all[:-1]
        # Without a blank, the offsets always immediately precede the onsets by one sample, 
        #     except the first onset, which is the first stimulus.
        sample_offsets = sample_onsets_all[1:] - 1
        # Get the number of samples in each pulse.
        sample_duration = (sample_offsets - sample_onsets) + 1
        # Get the onset and offset times (in seconds) relative to the start of the recording.
        time_onsets = self._get_sample_time(sample_onsets)
        time_offsets = self._get_sample_time(sample_offsets)
        # Get the duration of each pulse in seconds.
        time_duration = time_offsets - time_onsets
        # Get the value of each pulse by taking the mean of each range (onset, offset).
        value_array = np.zeros((sample_onsets.size,), dtype=int)
        for idx, (onset, offset) in enumerate(zip(sample_onsets, sample_offsets)):
            value = np.mean(self.signal[onset:offset+1])
            if (value == 0) | (value == 1):
                value_array[idx] = int(value)
            else:
                warnings.warn("Mean digital signal during a pulse is expected to be 0 or 1 but has a value of {0:f}.\n".format(value) + 
                                "    Assigning anyway, but this may indicate a problem with NI-DAQ digital data streams.")
                value_array[idx] = value

        # Add signal parameters to a dictionary:
        events_dict = {
            "sample_onset" :    sample_onsets,   # (# of samples to onset from sample 0, the start of recording)
            "sample_offset" :   sample_offsets,  # (# of samples to onset from sample 0, the start of recording)
            "sample_duration" : sample_duration, # (# of samples from onset to offset, inclusive)
            "time_onset" :      time_onsets,     # (seconds to sample onset, from time 0, the start of recording)
            "time_offset" :     time_offsets,    # (seconds to sample onset, from time 0, the start of recording)
            "time_duration" :   time_duration,   # (seconds from onset to offset, ???)
            "digital_value" :   value_array,     # (0 or 1, should alternate when there is no blank)
                        }
        # Create dataframe.
        self.events = pd.DataFrame.from_dict(events_dict, orient="columns")
        return None


    def _get_events(
            self, 
        ):
        """"""

        self._Events = []
        for row in self.events.itertuples():
            self._Events.append(Event(*row))
        return None


    def _get_sample_time(
            self, 
            sample_num, 
        ):
        """

        """

        sample_length = 1/self.sampling_rate
        sample_times_all = np.arange(0, self.total_time, sample_length, dtype=float)
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


    # def plot_digital_channels(
    #     self,
    #     signal_names=["sync", "camera", "pc_clock", "photodiode"],
    #     time_window=[0, 100], 
    #     ):
    #     """

    #     """
    
    #     mpl_use("Qt5Agg")
    #     fig, axs = plt.subplots(len(signal_names), 1)
    #     for n, signal_name in enumerate(signal_names):
    #         signal, timepoints, _ = self._get_digital_signals(signal_name, time_window)
    #         axs[n].plot(timepoints, signal.squeeze())
    #         axs[n].set_title(signal_name)
    #         axs[n].set_xlabel("Time (s)")
    #         axs[n].set_ylim((-0.1, 1.1))
    #         axs[n].set_yticks((0, 1))
    #         axs[n].set_ylabel("Signal Value")
    #     fig.set_size_inches(12, 6)
    #     plt.show()
    

    # def check_time_stability(
    #     self, 
    #     signal_name="sync", 
    #     blank=False, 
    #     plot=False, 
    #     ):
    #     """
    #     Check any digital signal's stability over time by comparing the number of samples encompassed by each pulse.

    #     This is particularly useful for checking the sampling rate over time using the sync signal (default).
    #         This signal should be a 1 Hz pulse with a 0.5 s duration.
    #         If the length of the pulse varies, the sampling rate is inconsistent across the recording.

    #     This is probably not useful for stimuli whose duration varies randomly. 
    #     """

    #     # Get the sync digital signal.
    #     line_num = self._parse_signal_IDs(signal_name)
    #     signal = SGLXReader.ExtractDigital(SGLXReader(), 
    #                                         self.bin_memmap, 
    #                                         0, self.total_samples-1, 
    #                                         0, 
    #                                         line_num, 
    #                                         self.meta_dict)
    #     # First, there is never a blank in the sync signal.
    #     assert (line_num == 7) != (blank == True), "There is never a blank in the sync signal, did you mean to specify blank=False?"
    #     # If there is no blank, both 0 and 1 indicate a stimulus.
    #     # In this case, interleave sample counts depending on starting point (0 or 1).
    #     if blank == False:
    #         labels = sk_label(signal[0])
    #         labels_a = np.abs(labels - signal[0, 0])
    #         _, counts_a = np.unique(labels - signal[0, 0], return_counts=True)
    #         labels_b = np.abs(labels - signal[0, 0])
    #         _, counts_b = np.unique(labels - np.abs(signal[0, 0]-1), return_counts=True)
    #         counts_full = np.zeros(counts_a.size + counts_b.size - 2, dtype=int)
    #         # Counting 1's in both cases, so leave out the 0 (first) count either way.
    #         counts_full[0::2] += counts_a[1:]
    #         counts_full[1::2] += counts_b[1:]
    #         # For these quality control purposes, we'll set the first and last counts to NaN.
    #         #     - the sync signal is almost always cut off at both ends, which is expected.
    #         #     - the first stimulus always (???) starts with a pulse to 1
    #         #     - the last stimulus always ends going back to zero
    #         # These sources of variation with squeeze the y-axis and preclude close examination.
    #         counts_full[(1, -1)] = (np.nan, np.nan)
    #     # If there is a blank, 1's indicate a stimulus, so just count these pulses.
    #     else:
    #         labels = sk_label(signal[0])
    #         _, counts = np.unique(labels, return_counts=True)
    #         counts_full = counts[1:]

    #     # If specified, make a line plot to show the sampling rate stability over the recording.
    #     if plot == True:
    #         #### Need to align X with center of pulses (not regionprops, maybe manual)
    #         X = np.linspace(0, self.total_time, counts_full.size)
    #         mpl_use("Qt5Agg")
    #         fig, ax = plt.subplots()
    #         ax.plot(X, counts_full)
    #         ax.set_title("Sampling Rate Stability over Recording")
    #         ax.set_xlabel("Time (s)")
    #         ax.set_ylabel("# of Samples in Sync Pulse")
    #         #### Add probability-normalized histogram?
    #         fig.set_size_inches(10, 4)
    #         plt.show()
    #     return counts_full #### Make this into a df with numbered index and 0/1 indicator column




class SyncChannel(DigitalChannel):

    def __init__(
            self, 
            signal, 
            sampling_rate, 
        ):
        """

        """

        super().__init__(signal, 
                         sampling_rate)
