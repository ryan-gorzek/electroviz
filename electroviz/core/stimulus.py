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

        self._Sync, self._PC_Clock, self._Photodiode = nidaq[:3]
        self._VStim = btss[0]


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

        # Stack photodiode and vstim dataframes.
        vstim_df = self._map_btss_vstim(self._VStim.events)
        self.events = pd.concat((self._Photodiode.events, vstim_df), axis=1)

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

    def __len__(self):
        return len(self._Events)

    # def plot_btss_vstim_times

    def _map_btss_vstim(
            self, 
            vstim_df, 
        ):
        """

        """

        # Define stimulus parameters from rig log to keep.
        param_names = ["contrast", "posx", "posy", "ori", "sf", "phase", "tf", "itrial", "istim"]
        # Initialize list for stacking visual stimulus parameters.
        params_list = []
        vstim_times = np.array(vstim_df["timereceived"])
        # Get the pc_clock dataframe for aligning bTsS trigger data.
        df_align = self._PC_Clock.events
        # Iterate through pc_clock pulses and find vstim entries that occur during the pulse.
        for (onset_time, offset_time) in df_align[["time_onset", "time_offset"]].to_numpy():
            vstim_logical = ((onset_time <= vstim_times) &
                                (vstim_times <= offset_time))
            if np.any(vstim_logical):
                params = self._capture_vstim(vstim_df, vstim_logical, param_names)
                params_list.append(params)
        # Create a dataframe from the visual stimuli parameters.
        vstim_df_mapped = pd.DataFrame(params_list, columns=param_names)
        return vstim_df_mapped

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

    def _get_events(
            self, 
            params, 
        ):
        """"""

        self._Events = []
        for row in self.events.itertuples():
            stim_indices = self._get_stim_indices(row[0], params=params)
            self._Events.append(Event(stim_indices, *row))
        return None

    def _get_stim_indices(
            self, 
            row_index, 
            params, 
        ):
        """"""

        values = tuple(self.events[params].iloc[row_index])
        indices = []
        for val, param in zip(values, params):
            (idx,) = np.where(np.unique(self.events[param]) == val)
            indices.append(idx[0])
        return tuple(indices)

    def _get_shape(
            self, 
            params, 
        ):
        """"""
        
        self.shape = tuple(np.unique(self.events[param]).size for param in params)
        return None




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

        self._get_events(params=["contrast", "posx", "posy", "itrial"])
        self._get_shape(params=["contrast", "posx", "posy", "itrial"])
        self._get_unique()

    def _get_unique(
            self, 
        ):
        """"""
        
        self.unique = []
        for contrast in sorted(np.unique(self.events["contrast"])):
            for posx in sorted(np.unique(self.events["posx"])):
                for posy in sorted(np.unique(self.events["posy"])):
                    self.unique.append((contrast, posx, posy))
        return None



    
class StaticGratings(VisualStimulus):
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

        self._get_events(params=["ori", "sf", "phase", "itrial"])
        self._get_shape(params=["ori", "sf", "phase", "itrial"])
        self._get_unique()

    def _get_unique(
            self, 
        ):
        """"""
        
        self.unique = []
        for ori in sorted(np.unique(self.events["ori"])):
            for sf in sorted(np.unique(self.events["sf"])):
                for phase in sorted(np.unique(self.events["phase"])):
                    self.unique.append((ori, sf, phase))
        return None


# class OptogeneticStimulus

# class ParallelStimulus

# class DriftingGratings(VisualStimulus):

# class ContrastReversal(VisualStimulus):

# class LightSquarePulse(OptogeneticStimulus):

# class LightSineWave(OptogeneticStimulus):
        