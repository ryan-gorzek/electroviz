# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.unit import Unit
import pandas as pd
import numpy as np
import copy
import xarray as xr
import matplotlib.pyplot as plt

class Population:
    '''
    docstring
    '''

    def __init__(self, parent, units_df, electrodes_df):
        print('Population')
        self._Units = []
        self.unit_ids = units_df.index.values
        self.info_df = pd.DataFrame()
        self.quality_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        # Create times-by-units array for spike times
        num_units = len(self.unit_ids)
        # max_spike_times = np.max(units_df[["spike_times"]].applymap(len).values)
        # self.spike_times = xr.DataArray(np.full((max_spike_times, num_units), np.nan), 
        #                                     dims = ("times", "units"), 
        #                                     coords = {"times":range(max_spike_times), 
        #                                               "units":self.unit_ids, 
        #                                               })
        # Create time-by-units-by-channels array for mean waveforms
        max_channels = np.max(electrodes_df["probe_id"].value_counts())
        time_series = np.linspace(0, 2.7, num=82)
        self.mean_waveforms = xr.DataArray(np.full((len(time_series), num_units, max_channels), np.nan), 
                                           dims = ("times", "units", "channels"), 
                                           coords = {"times":time_series, 
                                                     "units":self.unit_ids, 
                                                     "channels":range(max_channels), 
                                                     })
        for uid in self.unit_ids:
            curr_unit_df = units_df.loc[[uid], :]
            unit_probe_id = self._get_unit_probe_id(electrodes_df, curr_unit_df)
            unit_local_index = self._get_unit_local_index(electrodes_df, curr_unit_df)
            unit_location = self._get_unit_location(electrodes_df, curr_unit_df)
            curr_unit = Unit(curr_unit_df, unit_probe_id, unit_local_index, unit_location)
            self._Units.append(curr_unit)
            # get full dataframes for easy Population manipulation
            self.info_df = pd.concat([self.info_df, 
                                      curr_unit.info_df])
            self.quality_df = pd.concat([self.quality_df, 
                                         curr_unit.quality_df])
            self.stats_df = pd.concat([self.stats_df, 
                                            curr_unit.stats_df])
            # self.spike_times.loc[range(len(curr_unit.spike_times)),uid] = curr_unit.spike_times
            self.mean_waveforms.loc[:,uid,:] = curr_unit.mean_waveforms

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
        elif isinstance(parsed_index, (list, tuple, np.ndarray)):
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

    def plot_mean_waveform(self, channels="peak"):
        """"""
        if channels == "peak":
            channels = np.asarray(self.info_df["probe_channel_number"])
        unit_idx = xr.DataArray(range(len(self.unit_ids)), dims=["units"])
        channel_idx = xr.DataArray(channels, dims=["units"])
        time_series = np.linspace(0, 2.7, num=82)
        plt.plot(time_series, self.mean_waveforms[:, unit_idx, channel_idx])
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential ($\mu$V)")
        # add unit id, channel (w/ peak indication), and location to lower right
        # skip some of this when plotting from Population?
        fig = plt.gcf()
        fig.set_size_inches(4, 4)
        plt.tight_layout()
        ax = plt.gca()
        ax.set_aspect(1./ax.get_data_ratio())
        return ax
    
    # def query(self, statement):
    #     """"""
    #     full_array = pd.concat([self.info_df, self.quality_df, self.stats_df], axis=1)
    #     queried_unit_ids = full_array.query(statement).index.values
    #     return self[queried_unit_ids]
    
    # def clone(self, name="default"):
    #     """"""
    #     if name == "default":
    #         name = self.name + "_clone"
    #     self.parent.population_names.append(name)
    #     setattr(self.parent, name, copy.copy(self))
    
    # def split(self, this_name="default", that_name="default"):
    #     """"""
    #     if this_name == "default":
    #         this_name = self.name + "_split1"
    #     setattr(self.parent, this_name, copy.copy(self))
    #     if that_name == "default":
    #         that_name = self.name + "_split0"
    #     self.delete()
    #     self.from_population.rename(that_name)

    # def delete(self):
    #     """"""
    #     unit_ids = self.unit_ids
    #     from_unit_ids = self.from_population.unit_ids
    #     del_idx = np.where(np.isin(from_unit_ids, unit_ids))[0]
    #     self.from_population._Units = list(np.delete(np.array(self.from_population._Units), del_idx))
    #     self.from_population.unit_ids = np.delete(from_unit_ids, del_idx)
    #     self.from_population.info_df.drop(from_unit_ids[del_idx], inplace=True)
    #     self.from_population.quality_df.drop(from_unit_ids[del_idx], inplace=True)
    #     self.from_population.stats_df.drop(from_unit_ids[del_idx], inplace=True)
    #     if np.all(np.isin(from_unit_ids, unit_ids)):
    #         delattr(self.from_population.parent, self.from_population.name)
    
    # def rename(self, new_name):
    #     """"""
    #     self.clone(name=new_name)
    #     self[:].delete()
    
    def _get_unit_probe_id(self, electrodes_df, unit_df):
        """Get a unit's probe_id by finding the probe_id containing its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "probe_id"]
    
    def _get_unit_local_index(self, electrodes_df, unit_df):
        """Get a unit's local_index by finding the local_index of its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "local_index"]
    
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
        elif isinstance(index, (list, tuple, np.ndarray)) and np.all(np.array(index) < self.unit_ids.shape[0]):
            parsed_index = index
        elif isinstance(index, (list, tuple, np.ndarray)) and np.all(np.array(index) >= self.unit_ids.shape[0]):
            parsed_index = []
            for idx in index:
                parsed_index.append(np.where(self.unit_ids == idx)[0][0])
        return parsed_index
        