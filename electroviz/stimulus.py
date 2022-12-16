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
            # Parse the NI-DAQ first.
            # Get sync signal from the NI-DAQ, needed to align to neural data.
            # Keep all sync data but digital line number.
            self.sync_df = self._dict_to_df(
                               nidaq_obj.digital_signals["sync"], 
                               omit_keys=["line_num"], 
                               )
            # Get stimulus (pc_clock and photodiode) signals from the NI-DAQ.
            # This will be aligned to stimulus information.
            # Keep all data but digital line numbers.
            pc_clock_df = self._dict_to_df(
                                 nidaq_obj.digital_signals["pc_clock"], 
                                 omit_keys=["line_num"], 
                                 )
            photodiode_df = self._dict_to_df(
                                 nidaq_obj.digital_signals["photodiode"], 
                                 omit_keys=["line_num"], 
                                 )
            # Build a multi-index dataframe with top-level indices for pc_clock and photodiode.
            self.digital_df = self._build_multiindex_df(
                                    (pc_clock_df, photodiode_df), 
                                    ("pc_clock", "photodiode"), 
                                    )
            # Reference the NIDAQ object.
            self.nidaq_obj = nidaq_obj
        elif time_intervals_obj is not None:
            self.info_df = pd.DataFrame()
            self.info_df["stimulus_name"] = list(time_intervals_obj['stimulus_name'].data)
            self.info_df["start_time"] = np.array(time_intervals_obj['start_time'].data)
            self.info_df["stop_time"] = np.array(time_intervals_obj['stop_time'].data)
        else:
            raise Exception("Can't construct Stimulus, verify that data streams are specified correctly.")


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
            {key : val for (key, val) in dict_in.items() if key in keep_keys}, 
            orient="index", 
                                       )
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
                # Construct the top-level index from the specified name and the existing index names.
                num_idx = len(df.index)
                top_level_names = np.repeat(name, num_idx)
                df["event"] = top_level_names
                df.index.name = "param"
                # Add the top-level index to dataframe.
                df_multi = df.set_index(["event", df.index])
                dfs_multi.append(df_multi)
        # Concatenate now that dataframes are all multi-index.
        df_out = pd.concat(dfs_multi, axis=0)
        return df_out


class VisualStimulus(Stimulus):
    """

    """

    def __init__(
            self, 
            nidaq_obj=None, 
            btss_obj=None, 
            time_intervals_obj=None,
        ):

        super().__init__(
                    nidaq_obj, 
                    btss_obj, 
                    time_intervals_obj, 
                    )

        if btss_obj is None:
            raise Exception("Failed to build VisualStimulus, verify that data streams are specified correctly.")
        # Reference the bTsS object.
        self.btss_obj = btss_obj
        # Map the visual stimulus (vstim) dataframe from the bTsS rig log to the NIDAQ pulses.
        self.stim_df = self._map_btss_vstim(
                                btss_obj.riglog[0]["vstim"]
                            )

    # def plot_btss_vstim_times

    def _map_btss_vstim(
            self, 
            vstim_df, 
        ):
        """

        """

        # Initialize list for stacking visual stimulus parameters.
        params_list = []
        vstim_times = np.array(vstim_df["timereceived"])
        # Get the pc_clock dataframe for aligning bTsS trigger data.
        df_align = self.digital_df.loc[("photodiode"), :]
        # Iterate through pc_clock pulses and find vstim entries that occur during the pulse.
        for (onset_time, offset_time) in df_align.loc[["time_onsets", "time_offsets"]].T.to_numpy():
            vstim_logical = ((onset_time <= vstim_times) &
                                (vstim_times <= offset_time))
            if np.any(vstim_logical):
                self._verify_vstim_capture(vstim_df, vstim_logical)

    def _verify_vstim_capture(
            self,
            vstim_df, 
            vstim_logical, 
        ):
        # Thoroughly check for correct alignment of riglog-encoded visual stimulus and NI-DAQ.
        num_vstim_samples = np.sum(vstim_logical)
        vstim_params = vstim_df[["istim", "posx", "posy"]].values
        vstim_params_capture = vstim_params[vstim_logical, :]
        vstim_samples_match = all((vstim_params_capture == vstim_params_capture[[0], :]).all(axis=1))
        print(vstim_params_capture)
        print("{0} samples, {1} match".format(num_vstim_samples, vstim_samples_match))





# class OptogeneticStimulus

# class ParallelStimulus

# class SparseNoise(VisualStimulus):

# class Orientation(VisualStimulus):

# class SpatialFrequency(VisualStimulus):

# class TemporalFrequency(VisualStimulus):

# class ContrastReversal(VisualStimulus):

# class LightSquarePulse(OptogeneticStimulus):

# class LightSineWave(OptogeneticStimulus):
        