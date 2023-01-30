
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

class LFP:
    """
    
    """


    def __init__(
            self, 
            channels, 
            channel_positions, 
            sampling_rate, 
        ):
        """"""

        self.channels = channels
        self.channel_positions = channel_positions
        self.sampling_rate = sampling_rate
        self.total_samples = self.channels.shape[1]
        self.total_time = self.total_samples / self.sampling_rate

