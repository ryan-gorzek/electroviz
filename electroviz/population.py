# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Population:
    '''
    docstring
    '''
    
    def __init__(self, units_df, electrodes_df):
        print('Population')
        from electroviz.unit import Unit
        import pandas as pd
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
    
    def __getitem__(self, index_slice_or_unit_id):
        sub_self = self
        if isinstance(index_slice_or_unit_id, slice):
            sub_self._Units = self._Units[index_slice_or_unit_id]
            sub_self.unit_ids = self.unit_ids[index_slice_or_unit_id]
            sub_self.info_df = self.info_df.iloc[index_slice_or_unit_id]
            sub_self.quality_df = self.info_df.iloc[index_slice_or_unit_id]
            sub_self.stats_df = self.info_df.iloc[index_slice_or_unit_id]
        return sub_self
            
    def _get_unit_probe_id(self, electrodes_df, unit_df):
        """Get a unit's probe_id by finding the probe_id containing its peak_channel_id"""
        peak_channel_id = unit_df.at[unit_df.index[0], "peak_channel_id"]
        return electrodes_df.at[peak_channel_id, "probe_id"]
    

    
    # def split
        