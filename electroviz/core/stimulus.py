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
        vstim_events, event_idx = self._map_btss_vstim(self._VStim)
        photodiode_events = self._Photodiode.events.iloc[event_idx].reset_index()
        photodiode_events.drop(columns=["index"], inplace=True)
        self.events = pd.concat((photodiode_events, vstim_events), axis=1)

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
            vstim, 
        ):
        """

        """

        # Define stimulus parameters from rig log to keep.
        param_names = ["contrast", "posx", "posy", "ori", "sf", "phase", "tf", "itrial", "istim"]
        # Correct bTsS times for concatenation.
        if self._PC_Clock.concat_times is not None:
            concat_time = self._PC_Clock.concat_times[vstim.index]
        else:
            concat_time = 0.0
        concat_idx = np.where(self._PC_Clock.events["time_onset"] >= concat_time)[0][0]
        num_events = vstim.events.shape[0]
        event_idx = np.arange(concat_idx, concat_idx + num_events, 1)
        # Create a dataframe from the visual stimuli parameters.
        vstim_df_mapped = vstim.events[param_names]
        return vstim_df_mapped, event_idx

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
        