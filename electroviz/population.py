# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.unit import Unit
import pandas as pd
import numpy as np

class Population:
    '''
    docstring
    '''

    def __init__(self, units_df, electrodes_df):
        print('Population')
        self._Units = []
        self.unit_ids = units_df.index.values
        self.info_df = pd.DataFrame()
        self.quality_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        for uid in self.unit_ids:
            curr_unit_df = units_df.loc[[uid], :]
            unit_probe_id = self._get_unit_probe_id(electrodes_df, curr_unit_df)
            curr_unit = Unit(curr_unit_df, unit_probe_id)
            self._Units.append(curr_unit)
            # get full dataframes for easy Population manipulation
            self.info_df = pd.concat([self.info_df, 
                                      curr_unit.info_df])
            self.quality_df = pd.concat([self.quality_df, 
                                         curr_unit.quality_df])
            self.stats_df = pd.concat([self.stats_df, 
                                            curr_unit.stats_df])
    
    def __getitem__(self, index_slice_or_unit_ids):
        """"""
        parsed_index = self._parse_index(index_slice_or_unit_ids)
        if isinstance(parsed_index, slice):
            item = self.__copy__()
            item._Units = item._Units[parsed_index]
            item.unit_ids = item.unit_ids[parsed_index]
            item.info_df = item.info_df.iloc[parsed_index]
            item.quality_df = item.quality_df.iloc[parsed_index]
            item.stats_df = item.stats_df.iloc[parsed_index]
        elif isinstance(parsed_index, (list, tuple)):
            item = self.__copy__()
            item._Units = [item._Units[idx] for idx in parsed_index]
            item.unit_ids = item.unit_ids[parsed_index]
            item.info_df = item.info_df.iloc[parsed_index]
            item.quality_df = item.quality_df.iloc[parsed_index]
            item.stats_df = item.stats_df.iloc[parsed_index]
        else:
            item = self._Units[parsed_index]
            
            
    def __copy__(self):
        import copy
        return copy.copy(self)
            
    def _get_unit_probe_id(self, electrodes_df, unit_df):
        """Get a unit's probe_id by finding the probe_id containing its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "probe_id"]
    
    def _parse_index(self, index):
        if isinstance(index, slice):
            parsed_index = index
        elif isinstance(index, int) and index < self.unit_ids.shape[0]:
            parsed_index = index
        elif isinstance(index, int) and index >= self.unit_ids.shape[0]:
            parsed_index = np.where(self.unit_ids == index)[0][0]
        elif isinstance(index, (list, tuple)) and np.all(np.array(index) < self.unit_ids.shape[0]):
            parsed_index = index
        elif isinstance(index, (list, tuple)) and np.all(np.array(index) >= self.unit_ids.shape[0]):
            parsed_index = []
            for idx in index:
                parsed_index.append(np.where(self.unit_ids == idx)[0][0])
        return parsed_index
    
    # def split
        