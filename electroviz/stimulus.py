# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd

class Stimulus:
    '''
    docstring
    '''
    
    def __init__(self, time_intervals_object):
        print('Stimulus')
        self.info_df = pd.DataFrame()
        start_times = time_intervals_object['start_time'].data
        stop_times = time_intervals_object['stop_time'].data
        num_times = start_times.shape[0]
        for ts in range(num_times):
            self.info_df = pd.concat([self.info_df,
                                      pd.DataFrame(index=[ts],
                                                   data={'start_time':start_times[ts],
                                                         'stop_time':stop_times[ts]})])
        