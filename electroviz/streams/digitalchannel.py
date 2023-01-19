# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
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
            concat_times=None, 
        ):
        """"""

        # Get some basic data and parameters for easy access.
        self.signal = signal.squeeze()
        self.sampling_rate = sampling_rate
        self.concat_times = concat_times
        self.total_samples = len(self.signal)
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
        """"""

        # Remove specified samples from the digital signal.
        self.signal = np.delete(self.signal, drop_samples)
        # Update signal parameters.
        self.total_samples = len(self.signal)
        self.total_time = self.total_samples / self.sampling_rate
        # Update the event times dataframe and list of Event objects.
        self._build_events_df()
        self._get_events()
        # Check for single-sample events and flip them.
        if any(self.events["sample_duration"] == 1):
            (idx,) = np.where(self.events["sample_duration"] == 1)
            sample_index = self.events.at[idx[0], "sample_onset"]
            digital_value = self.events.at[idx[0], "digital_value"]
            if digital_value == 0:
                self.signal[sample_index] = 1
            elif digital_value == 1:
                self.signal[sample_index] = 0
            self._build_events_df()
            self._get_events()


    def _build_events_df(
            self, 
        ):
        """"""

        # Get value for digital signal at recording start and end.
        value_start = self.signal[0]
        value_end = self.signal[-1]
        # Take temporal difference of digital signal for locating onsets and offsets.
        signal_diff = np.insert(np.diff(self.signal), 0, 0)
        # A stimulus should start 0->1, then flip up-and-down, making any non-zero value of the diff an onset, 
        #     except the last.
        (sample_onsets_all,) = np.nonzero(signal_diff != 0)
        # The last "onset" (non-zero value) is really the first sample after the final stimulus, so we drop it.
        sample_onsets = sample_onsets_all
        # Without a blank, the offsets always immediately precede the onsets by one sample, 
        #     except the first onset, which is the first stimulus.
        sample_offsets = np.append(sample_onsets_all[1:] - 1, [self.signal.size - 1], axis=0)
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
        """"""

        sample_length = 1/self.sampling_rate
        sample_times_all = np.arange(0, self.total_time, sample_length, dtype=float)
        sample_times = sample_times_all[sample_num]
        return sample_times


    def _get_time_sample(
            self, 
            time_window=(0, None), 
            return_range=True, 
        ):
        """"""

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




class SyncChannel(DigitalChannel):
    """

    """


    def __init__(
            self, 
            signal, 
            sampling_rate, 
            concat_times=None, 
        ):
        """

        """

        super().__init__(signal, 
                         sampling_rate, 
                         concat_times=None)

