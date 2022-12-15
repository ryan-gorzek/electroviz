# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np

class Stimulus:
    """

    """
    
    def __init__(
            self, 
            nidaq_obj=None, 
            btss_obj=None, 
            time_intervals_obj=None, 
        ):
        """
        
        """

        if nidaq_obj is not None:
            # Parse the NI-DAQ first. Map the digital signal keys to dataframe column names.
            #### This might be kind of pointless.
            names_map = {"sample_onsets" : "onset_sample", 
                         "sample_offsets" : "offset_sample", 
                         "sample_durations" : "duration_sample", 
                         "time_onsets" : "onset_time", 
                         "time_offsets" : "offset_time", 
                         "time_durations" : "duration_time", 
                         "digital_value" : "digital_value"
                        }
            column_order = ["onset_sample", "offset_sample", "duration_sample", 
                            "onset_time", "offset_time", "duration_time", 
                            "digital_value"]
            # Get sync signal from the NI-DAQ, needed to align to neural data.
            # Keep all sync data but digital line number.
            self.sync_df = self._dict_to_df(
                               nidaq_obj.digital_signals["sync"], 
                               omit_keys=["line_num"], 
                               names_map=names_map, 
                               column_order=column_order, 
                               )
            # Get stimulus (pc_clock and photodiode) signals from the NI-DAQ.
            # This will be aligned to stimulus information.
            # Keep all data but digital line numbers.
            pc_clock_df = self._dict_to_df(
                                 nidaq_obj.digital_signals["pc_clock"], 
                                 omit_keys=["line_num"], 
                                 names_map=names_map, 
                                 column_order=column_order, 
                                 )
            photodiode_df = self._dict_to_df(
                                 nidaq_obj.digital_signals["photodiode"], 
                                 omit_keys=["line_num"], 
                                 names_map=names_map, 
                                 column_order=column_order, 
                                 )
            # Build a multi-index dataframe with top-level indices for pc_clock and photodiode.
            self.stim_df = self._build_multiindex_df(
                                    (pc_clock_df, photodiode_df), 
                                    ("pc_clock", "photodiode"), 
                                    )
        elif time_intervals_obj is not None:
            self.info_df = pd.DataFrame()
            self.info_df["stimulus_name"] = list(time_intervals_obj['stimulus_name'].data)
            self.info_df["start_time"] = np.array(time_intervals_obj['start_time'].data)
            self.info_df["stop_time"] = np.array(time_intervals_obj['stop_time'].data)
        else:
            raise Exception("Failed to build Stimulus, verify that data streams are specified correctly.")


    def _dict_to_df(
            self,
            dict_in, 
            omit_keys=None,
            names_map=None, 
            column_order=None, 
        ):
        """

        """

        keep_keys = set(dict_in.keys()).difference(omit_keys)
        df_out = pd.DataFrame.from_dict(
            {key : val for (key, val) in dict_in.items() if key in keep_keys}
                                       )
        if names_map is not None:
            df_out.rename(columns=names_map)
        # if column_order is not None:
        #     df_out = df_out.reindex(column_order, axis=1)
        return df_out

    def _build_multiindex_df(
            self,
            dfs_in, 
            df_names=None, 
        ):
        """

        """

        # Append all multi-index dataframes (converted or passed) to list, concatenate later.
        dfs_multi = []
        for (df, name) in zip(dfs_in, df_names):
            # Check whether dataframe was already multi-index when passed.
            if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
                dfs_multi.append(df)
            else:
                # Construct the top-level index from the specified name and the existing column names.
                num_columns = len(df.columns)
                top_level_names = np.repeat(name, num_columns)
                # Build the multi-index dataframe.
                columns = [top_level_names, np.array(df.columns)]
                df_converted = pd.DataFrame(df, columns=columns)
                dfs_multi.append(df_converted)
        # Concatenate now that dataframes are all multi-index.
        df_out = pd.concat(dfs_multi, axis=1)
        return df_out



class VisualStimulus

# class OptogeneticStimulus

# class ParallelStimulus

# class SparseNoise:

# class Orientation:

# class SpatialFrequency:

# class TemporalFrequency:

# class ContrastReversal:

# class LightSquarePulse:

# class LightSineWave:
        