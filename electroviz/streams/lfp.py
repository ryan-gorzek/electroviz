
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
            sampling_rate, 
        ):
        """"""

        self.channels = channels[:384, :]
        self.sampling_rate = sampling_rate
        self.total_samples = self.channels.shape[1]
        self.total_time = self.total_samples / self.sampling_rate
        

    def drop_and_rebuild(
            self, 
            drop_samples, 
        ):
        """"""

        # Create column (sample) logical index from sample indices.
        col_mask = np.ones(self.total_samples, dtype=bool)
        col_mask[drop_samples] = False
        self.total_samples += -len(drop_samples)
        self.channels = self.channels[:, col_mask]
        return None

