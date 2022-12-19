# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np
from warnings import warn

class Stimulus:
    """

    """
    
    def __init__(
            self, 
            sync=None, 
            pc_clock=None, 
            photodiode=None, 
            btss_obj=None, 
            time_intervals=None, 
        ):
        """
        
        """

        if sync is None:
            raise Exception("Stimulus requires a sync signal for alignment with neural data.")
        elif btss_obj is None:
            raise Exception("Stimulus requires a bTsS object for proper stimulus identification.")
        elif photodiode is None:
            raise Exception("Stimulus requires a photodiode signal for alignment with bTsS stimulus information.")
        elif time_intervals is not None:
            self.info_df = pd.DataFrame()
            self.info_df["stimulus_name"] = list(time_intervals_obj['stimulus_name'].data)
            self.info_df["start_time"] = np.array(time_intervals_obj['start_time'].data)
            self.info_df["stop_time"] = np.array(time_intervals_obj['stop_time'].data)
        # else:
        #     raise Exception("Can't construct Stimulus, verify that data streams are specified correctly.")

        self.sync = sync
        self.pc_clock = pc_clock
        self.photodiode = photodiode

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
            sync=None, 
            pc_clock=None, 
            photodiode=None, 
            btss_obj=None, 
            time_intervals=None, 
        ):

        super().__init__(
                    sync=sync, 
                    pc_clock=pc_clock, 
                    photodiode=photodiode, 
                    btss_obj=btss_obj, 
                    time_intervals=time_intervals, 
                    )

        self.btss_visprot = btss_obj.visprot
        self.btss_riglog = btss_obj.riglog
        self.btss_vstim_df = btss_obj.vstim_df
        # Get the pc_clock and photodiode dataframes.
        self.pc_clock_df = self.pc_clock.digital_df
        self.photodiode_df = self.photodiode.digital_df
        # Map the stimulus information from bTsS to the pc_clock signal from the NI-DAQ.
        self.vstim_df = self._map_btss_vstim(self.btss_vstim_df)
        # Stack photodiode and vstim dataframes.
        self.stimulus_df = pd.concat((self.photodiode_df, self.vstim_df))

    # def plot_btss_vstim_times

    def _map_btss_vstim(
            self, 
            btss_vstim_df, 
        ):
        """

        """

        # Define stimulus parameters from rig log to keep.
        param_names = ["istim", "itrial", "contrast", "ori", "posx", "posy", "W", "H", "phase", "tf", "sf"]
        # Initialize list for stacking visual stimulus parameters.
        params_list = []
        vstim_times = np.array(btss_vstim_df.loc["timereceived"])
        # Get the pc_clock dataframe for aligning bTsS trigger data.
        df_align = self.pc_clock_df
        # Iterate through pc_clock pulses and find vstim entries that occur during the pulse.
        for (onset_time, offset_time) in df_align.loc[["time_onset", "time_offset"]].T.to_numpy():
            vstim_logical = ((onset_time <= vstim_times) &
                                (vstim_times <= offset_time))
            if np.any(vstim_logical):
                params = self._capture_vstim(btss_vstim_df, vstim_logical, param_names)
                params_list.append(params)
        # Create a dataframe from the visual stimuli parameters.
        vstim_df = pd.DataFrame(params_list, columns=param_names).T
        return vstim_df

    def _capture_vstim(
            self,
            vstim_df, 
            vstim_logical, 
            param_names, 
        ):
        """"""

        # Thoroughly check for correct alignment of riglog-encoded visual stimulus and NI-DAQ pc_clock signal.
        vstim_params = vstim_df.loc[param_names].values
        vstim_params_capture = vstim_params[:, vstim_logical]
        vstim_samples_match = all((vstim_params_capture == vstim_params_capture[:, [0]]).all(axis=1))
        # if vstim_samples_match == False:
        #     warn("A stimulus from the bTsS rig log failed to match the NI-DAQ pc_clock signal.")
        # Return the "unique" stimulus parameters for a given pc_clock pulse, but all should match.
        # vstim_params_unique = np.unique(np.array(vstim_params_capture), axis=1).squeeze()
        vstim_params = np.array(vstim_params_capture)[:, 8]
        return vstim_params




class SparseNoise(VisualStimulus):
    """

    """

    def __init__(
            self, 
            sync=None, 
            pc_clock=None, 
            photodiode=None, 
            btss_obj=None, 
            time_intervals=None, 
        ):

        super().__init__(
                    sync=sync, 
                    pc_clock=pc_clock, 
                    photodiode=photodiode, 
                    btss_obj=btss_obj, 
                    time_intervals=time_intervals, 
                    )

        # Grab only the stimulus parameters relevant for sparse noise.
        self.stimulus_df.drop(index=["ori", "phase", "tf", "sf"], inplace=True)
        # Build iterable.
        self._iterable = []
        posx_unique = np.unique(self.stimulus_df.loc["posx"].to_numpy())
        posy_unique = np.unique(self.stimulus_df.loc["posy"].to_numpy())
        contrast_unique = np.unique(self.stimulus_df.loc["contrast"].to_numpy())
        for _, stim in self.stimulus_df.iteritems():
            sd = stim.to_dict()
            sets = (int(sd["sample_onset"]), int(sd["sample_offset"]))
            posx_idx = int(np.where(posx_unique == sd["posx"])[0])
            posy_idx = int(np.where(posy_unique == sd["posy"])[0])
            contrast_idx = int(np.where(contrast_unique == sd["contrast"])[0])
            params = (posx_idx, posy_idx, contrast_idx)
            self._iterable.append((sets, params))
        self._current_stim_idx = 0

    def __iter__(self):
        return iter(self._iterable)

    def __next__(self):
        """"""
        if self._current_stim_idx < len(self._iterable):
            stim = self._iterable[self._current_stim_idx]
            self._current_stim_idx += 1
            return stim

# class OptogeneticStimulus

# class ParallelStimulus

# class StaticGratings(VisualStimulus):

# class DriftingGratings(VisualStimulus):

# class ContrastReversal(VisualStimulus):

# class LightSquarePulse(OptogeneticStimulus):

# class LightSineWave(OptogeneticStimulus):
        