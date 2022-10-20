# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.unit import Unit
import pandas as pd
import numpy as np
import copy

class Population:
    '''
    docstring
    '''

    def __init__(self, parent, units_df, electrodes_df, population_name):
        print('Population')
        self.name = population_name
        self.from_population = None
        self.parent = parent
        self._Units = []
        self.unit_ids = units_df.index.values
        self.info_df = pd.DataFrame()
        self.quality_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        for uid in self.unit_ids:
            curr_unit_df = units_df.loc[[uid], :]
            unit_probe_id = self._get_unit_probe_id(electrodes_df, curr_unit_df)
            unit_location = self._get_unit_location(electrodes_df, curr_unit_df)
            curr_unit = Unit(curr_unit_df, unit_probe_id, unit_location)
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
            item = copy.copy(self)
            item._Units = item._Units[parsed_index]
            item.unit_ids = item.unit_ids[parsed_index]
            item.info_df = item.info_df.iloc[parsed_index]
            item.quality_df = item.quality_df.iloc[parsed_index]
            item.stats_df = item.stats_df.iloc[parsed_index]
        elif isinstance(parsed_index, (list, tuple)):
            item = copy.copy(self)
            item._Units = [item._Units[idx] for idx in parsed_index]
            item.unit_ids = item.unit_ids[parsed_index]
            item.info_df = item.info_df.iloc[parsed_index]
            item.quality_df = item.quality_df.iloc[parsed_index]
            item.stats_df = item.stats_df.iloc[parsed_index]
        else:
            item = self._Units[parsed_index]
        item.from_population = self
        return item

    def plot_mean_waveforms(self, channels="peak", colors=[[0.7,0.2,0.2],[0.2,0.7,0.2],[0.2,0.2,0.7],[0.7,0.2,0.7]]):
        if channels == "peak":
            channels = [channels]*self.info_df.shape[0]
        color_idx = np.tile(np.arange(len(colors)), (1, int(np.ceil(self.info_df.shape[0]/len(colors)))))[0]
        for unit_num,unit in enumerate(self._Units):
            ax = unit.plot_mean_waveform(channel=channels[unit_num], color=colors[color_idx[unit_num]])
        return ax
    
    # def filter()
    
    def clone(self, name="default"):
        """"""
        if name == "default":
            name = self.name + "_clone"
        self.parent.population_names.append(name)
        setattr(self.parent, name, copy.copy(self))
    
    def split(self, this_name="default", rest_name="default"):
        """"""
        if this_name == "default":
            this_name = self.name + "_split1"
        setattr(self.parent, this_name, copy.copy(self))
        if rest_name == "default":
            rest_name = self.name + "_split0"
        # self.from_population.rename(rest_name)
        self.delete()
        
    def delete(self):
        """"""
        unit_ids = self.unit_ids
        from_unit_ids = self.from_population.unit_ids
        del_idx = np.where(np.isin(from_unit_ids, unit_ids))[0]
        self.from_population._Units = list(np.delete(np.array(self.from_population._Units), del_idx))
        self.from_population.unit_ids = np.delete(from_unit_ids, del_idx)
        self.from_population.info_df.drop(from_unit_ids[del_idx], inplace=True)
        self.from_population.quality_df.drop(from_unit_ids[del_idx], inplace=True)
        self.from_population.stats_df.drop(from_unit_ids[del_idx], inplace=True)
    
    # def rename()
    
    def _get_unit_probe_id(self, electrodes_df, unit_df):
        """Get a unit's probe_id by finding the probe_id containing its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "probe_id"]
    
    def _get_unit_location(self, electrodes_df, unit_df):
        """Get a unit's location by finding the location of its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "location"]
    
    def _parse_index(self, index):
        """"""
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
        