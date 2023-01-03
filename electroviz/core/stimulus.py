# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np
from warnings import warn
from electroviz.core.event import Event

class Stimulus:
    """

    """
    
    def __init__(
            self, 
            nidaq=None, 
            btss=None, 
        ):
        """
        
        """

        self.sync, self.pc_clock, self.photodiode = nidaq[:3]
        self.vstim = btss[0]


class VisualStimulus(Stimulus):
    """

    """

    def __init__(
            self, 
            nidaq=None, 
            btss=None, 
        ):

        super().__init__(
                         nidaq=nidaq, 
                         btss=btss, 
                        )

        self.btss_vstim_df = self.vstim.events
        # Get the pc_clock and photodiode dataframes.
        self.pc_clock_df = self.pc_clock.events
        self.photodiode_df = self.photodiode.events
        # Map the stimulus information from bTsS to the pc_clock signal from the NI-DAQ.
        self.vstim_df = self._map_btss_vstim(self.btss_vstim_df)
        # Stack photodiode and vstim dataframes.
        self.stimulus_df = pd.concat((self.photodiode_df, self.vstim_df), axis=1)

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
        vstim_times = np.array(btss_vstim_df["timereceived"])
        # Get the pc_clock dataframe for aligning bTsS trigger data.
        df_align = self.pc_clock_df
        # Iterate through pc_clock pulses and find vstim entries that occur during the pulse.
        for (onset_time, offset_time) in df_align[["time_onset", "time_offset"]].to_numpy():
            vstim_logical = ((onset_time <= vstim_times) &
                                (vstim_times <= offset_time))
            if np.any(vstim_logical):
                params = self._capture_vstim(btss_vstim_df, vstim_logical, param_names)
                params_list.append(params)
        # Create a dataframe from the visual stimuli parameters.
        vstim_df = pd.DataFrame(params_list, columns=param_names)
        return vstim_df

    def _capture_vstim(
            self,
            vstim_df, 
            vstim_logical, 
            param_names, 
        ):
        """"""

        # Thoroughly check for correct alignment of riglog-encoded visual stimulus and NI-DAQ pc_clock signal.
        vstim_params = vstim_df[param_names].values
        vstim_params_capture = vstim_params[vstim_logical, :]
        vstim_samples_match = all((vstim_params_capture == vstim_params_capture[[0], :]).all(axis=0))
        # if vstim_samples_match == False:
        #     warn("A stimulus from the bTsS rig log failed to match the NI-DAQ pc_clock signal.")
        # Return the "unique" stimulus parameters for a given pc_clock pulse, but all should match.
        # vstim_params_unique = np.unique(np.array(vstim_params_capture), axis=1).squeeze()
        vstim_params = np.array(vstim_params_capture)[8, :]
        return vstim_params




class SparseNoise(VisualStimulus):
    """

    """

    def __init__(
            self, 
            nidaq=None, 
            btss=None, 
        ):

        super().__init__(
                         nidaq=nidaq, 
                         btss=btss, 
                        )

        # Grab only the stimulus parameters relevant for sparse noise.
        self.events = self.stimulus_df.drop(columns=["ori", "phase", "tf", "sf"])
        self._get_events()


    def __getitem__(
            self, 
            index, 
        ):
        """"""
        return self._Events[index]

    
    def __iter__(
            self, 
        ):
        """"""
        self._Events_num = 0
        return iter(self._Events)


    def __next__(
            self, 
        ):
        """"""
        
        if self._Events_num > len(self._Events):
            raise StopIteration
        else:
            event = self.Events[self._Events_num]
            self._Events_num += 1
        return event


    def _get_events(
            self, 
        ):
        """"""

        self._Events = []
        for row in self.events.itertuples():
            self._Events.append(Event(*row))
        return None

# class OptogeneticStimulus

# class ParallelStimulus

# class StaticGratings(VisualStimulus):

# class DriftingGratings(VisualStimulus):

# class ContrastReversal(VisualStimulus):

# class LightSquarePulse(OptogeneticStimulus):

# class LightSineWave(OptogeneticStimulus):
        