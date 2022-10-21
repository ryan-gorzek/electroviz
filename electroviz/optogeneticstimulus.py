# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.stimulus import Stimulus
import numpy as np

class OptogeneticStimulus(Stimulus):
    '''
    docstring
    '''
    
    def __init__(self, time_intervals_obj):
        """"""
        print('OptogeneticStimulus')
        super().__init__(time_intervals_obj)
        self.info_df.insert(1, "condition", list(time_intervals_obj['condition'].data))
        self.info_df["duration"] = np.array(time_intervals_obj['duration'].data)
        self.info_df["level"] = np.array(time_intervals_obj['duration'].data)
        