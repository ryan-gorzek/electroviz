
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

class Probe:
    """
    
    """


    def __init__(
            self, 
            lfp, 
            sync, 
        ):
        """"""

        self._Sync = sync
        self.channels = lfp.channels
        self.sampling_rate = lfp.sampling_rate
        self.total_samples = lfp.total_samples
        self.total_time = lfp.total_samples / lfp.sampling_rate
        self.total_channels = self.channels.shape[0]
        # Define current index for iteration.
        self._current_channel_idx = 0

    def __getitem__(
            self, 
            input, 
        ):
        """"""

        try:
            subset = self.channels[input, :]
        except TypeError:
            print("Failed.")
        return subset


    def __iter__(self):
        """"""

        return iter(self.channels)


    def __next__(self):
        """"""

        if self._current_channel_idx < self.total_channels:
            channel = self.channels[self._current_channel_idx, :]
            self._current_channel_idx += 1
            return channel


    def __len__(self):
        """"""

        return self.channels.shape[0]


    def get_response(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
        ):
        """"""
        
        stimulus = stimulus.lfp()
        sample_window = np.array(time_window) * 2.5
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size * 2.5))
        responses = np.zeros((len(self), num_bins, len(stimulus)))
        for idx, event in enumerate(stimulus):
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.channels[:, window[0]:window[1]].toarray()
            mean_resp = resp.reshape((len(self), num_bins, -1)).mean(axis=2)
            responses[:, :, idx] = mean_resp
        return responses.mean(axis=2)

