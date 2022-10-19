# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np

class Stimulus:
    '''
    docstring
    '''
    
    def __init__(self, time_intervals_obj):
        print('Stimulus')
        self.info_df = pd.DataFrame()
        self.info_df["stimulus_name"] = list(time_intervals_obj['stimulus_name'].data)
        self.info_df["start_time"] = np.array(time_intervals_obj['start_time'].data)
        self.info_df["stop_time"] = np.array(time_intervals_obj['stop_time'].data)
        