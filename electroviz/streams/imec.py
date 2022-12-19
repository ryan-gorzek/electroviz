# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy import sparse
from electroviz.utils.extractDigital import extractDigital
import pandas as pd

class Imec:
    """

    """

    def __init__(
            self, 
            imec_metadata, 
            imec_binary, 
            kilosort_array, 
        ):

        # Get some basic data and parameters for easy access.
        self.imec_metadata = imec_metadata
        self.imec_binary = imec_binary
        self.sampling_rate = float(self.imec_metadata["imSampRate"])
        self.total_time = float(self.imec_metadata["fileTimeSecs"])
        self.total_samples = int(self.sampling_rate * self.total_time)

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




# class ImecProbe(Imec):
#     """

#     """

#     def __init__(
#             self, 
#             imec_metadata, 
#             imec_binary, 
#             kilosort_array, 
#         ):




class ImecSpikes(Imec):
    """
    
    """

    def __init__(
            self, 
            imec_metadata, 
            imec_binary, 
            kilosort_array, 
        ):

        super().__init__(imec_metadata, 
                         imec_binary, 
                         kilosort_array)

        # Get Kilosort data from columns of kilosort_array.
        (spike_clusters, spike_times) = np.hsplit(kilosort_array.flatten(), 2)
        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = int(np.max(spike_clusters) + 1)
        # Build (sparse) spike times matrix.
        self.spike_times = self._build_spike_times_matrix(spike_clusters, spike_times)

    def _build_spike_times_matrix(
            self, 
            spike_clusters, 
            spike_times, 
        ):
        """"""

        # Create a units-by-samples scipy sparse coordinate matrix to store spike times.
        full_shape = (self.total_units, self.total_samples)
        row_idx = spike_clusters
        col_idx = spike_times
        data = np.ones((spike_times.size,))
        spike_times_matrix = sparse.coo_matrix((data, (row_idx, col_idx)), shape=full_shape)
        return spike_times_matrix


class ImecSync(Imec):
    """

    """

    def __init__(
            self, 
            imec_metadata, 
            imec_binary, 
            kilosort_array, 
        ):
        """

        """

        super().__init__(imec_metadata, 
                         imec_binary, 
                         kilosort_array)
        # Add parameters specific to digital signals.
        self.line_number = 6
        self.digital_signal = extractDigital(self.imec_binary, 
                                             0, self.total_samples-1, 
                                             0, 
                                             [self.line_number], 
                                             self.imec_metadata).squeeze()
        self.digital_df = self._get_digital_times(blank=False)

    def _get_digital_times(
        self, 
        blank=False, 
        ):
        """
        
        """

        # Get value for digital signal at recording start and end.
        value_start = self.digital_signal[0]
        value_end = self.digital_signal[-1]
        # Get onsets and offsets depending on whether there is a blank.
        if blank == True:
            # Take temporal difference of digital signal for locating onsets and offsets.
            signal_diff = np.insert(np.diff(-1 * self.digital_signal), 0, 0)
            value_start, value_end = value_end, value_start
            (sample_onsets,) = np.where(signal_diff == 1)
            (sample_offsets,) = np.where(signal_diff == -1)
            sample_offsets += -1
        elif blank == False:
            # Take temporal difference of digital signal for locating onsets and offsets.
            signal_diff = np.insert(np.diff(self.digital_signal), 0, 0)
            # A stimulus without a blank should start 0->1, then flip up-and-down, 
            #     making any non-zero value of the diff an onset, except the last.
            (sample_onsets_all,) = np.where(signal_diff != 0)
            # The last "onset" (non-zero value) is really the first sample after the final stimulus, 
            #     so we drop it.
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
        value_array = np.zeros((sample_onsets.size,))
        for idx, (onset, offset) in enumerate(zip(sample_onsets, sample_offsets)):
            value = np.mean(self.digital_signal[onset:offset+1])
            if (value == 0) | (value == 1):
                value_array[idx] = int(value)
            else:
                warnings.warn("Mean digital signal during a pulse is expected to be 0 or 1 but has a value of {0:f}.\n".format(value) + 
                                "    Assigning anyway, but this may indicate a problem with NI-DAQ digital data streams.")
                value_array[idx] = value

        # Add signal parameters to a dictionary:
        params_dict = {
            "sample_onset" :    sample_onsets,   # (# of samples to onset from sample 0, the start of recording)
            "sample_offset" :   sample_offsets,  # (# of samples to onset from sample 0, the start of recording)
            "sample_duration" : sample_duration, # (# of samples from onset to offset, inclusive)
            "time_onset" :      time_onsets,     # (seconds to sample onset, from time 0, the start of recording)
            "time_offset" :     time_offsets,    # (seconds to sample onset, from time 0, the start of recording)
            "time_duration" :   time_duration,   # (seconds from onset to offset, ???)
            "digital_value" :    value_array,    # (0 or 1, should alternate when there is no blank)
                        }
        # Create dataframe.
        signals_df = pd.DataFrame.from_dict(params_dict, orient="index")
        return signals_df
