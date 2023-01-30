
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np
import copy
from warnings import warn
from electroviz.core.event import Event
import copy
import glob
from PIL import Image

class Stimulus:
    """

    """


    def __init__(
            self, 
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        self._Sync, self._PC_Clock, self._Photodiode = nidaq_ap[:3]
        if nidaq_lf is not None:
            self._lf = True
            self._Sync_, self._PC_Clock_, self._Photodiode_ = nidaq_lf[:3]
        else:
            self._lf = False
        self._VStim = btss[0]

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
        """"""

        return len(self._Events)


    def lfp(
            self, 
        ):
        """"""

        if self._lf is True:
            swapped = copy.copy(self)
            swapped._Sync, swapped._Sync_ = self._Sync_, self._Sync
            swapped._PC_Clock, swapped._PC_Clock_ = self._PC_Clock_, self._PC_Clock
            swapped._Photodiode, swapped._Photodiode_ = self._Photodiode_, self._Photodiode
            swapped._Events, swapped._Events_ = self._Events_, self._Events
            swapped.events, swapped._events_ = self._events_, self.events
        return swapped




class VisualStimulus(Stimulus):
    """

    """


    def __init__(
            self, 
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
                        )

        # Stack photodiode and vstim dataframes for AP.
        vstim_events = self._map_btss_vstim(self._VStim)
        self.events = pd.concat((self._Photodiode.events, vstim_events), axis=1)

        # Stack photodiode and vstim dataframes for LF.
        if nidaq_lf is not None:
            vstim_events = self._map_btss_vstim(self._VStim)
            self._events_ = pd.concat((self._Photodiode_.events, vstim_events), axis=1)

    def _map_btss_vstim(
            self, 
            vstim, 
        ):
        """"""

        # Define stimulus parameters from rig log to keep.
        param_names = ["contrast", "posx", "posy", "ori", "sf", "phase", "tf", "itrial", "istim"]
        vstim_df_mapped = vstim.events[param_names]
        return vstim_df_mapped


    def _get_events(
            self, 
            params, 
        ):
        """"""

        self._Events = []
        for row in self.events.itertuples():
            stim_indices = self._get_stim_indices(row[0], params=params)
            self._Events.append(Event(stim_indices, *row))
        # LFP.
        self._Events_ = []
        for row in self._events_.itertuples():
            stim_indices = self._get_stim_indices(row[0], params=params)
            self._Events_.append(Event(stim_indices, *row))
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
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
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
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
                        )

        self._get_events(params=["ori", "sf", "phase", "itrial"])
        self._get_shape(params=["ori", "sf", "phase", "itrial"])
        self._get_unique()
        self._get_gratings()


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

    
    def _get_gratings(
            self, 
            path="E:/random_gratings/*.png", 
        ):
        """"""

        gratings_files = glob.glob(path)
        self.gratings = []
        for grat in gratings_files[1::8]:
            img = Image.open(grat).convert("RGBA")
            self.gratings.append(np.array(img))
        self.gratings = np.array(self.gratings)




class ContrastReversal(VisualStimulus):
    """

    """


    def __init__(
            self, 
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        btss = self._match_vstim_df(btss)

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
                        )

        self._get_events(params=["contrast"])
        self._get_shape(params=["contrast"])
        self._get_unique()


    def _get_unique(
            self, 
        ):
        """"""
        
        self.unique = []
        for contrast in sorted(np.unique(self.events["contrast"])):
                    self.unique.append((contrast))
        return None


    def _match_vstim_df(
            self, 
            btss, 
        ):
        """"""

        btss_out = btss
        btss_out[0].events["contrast"] = btss[0].events["indicator"]
        for col in ["ori", "sf", "phase", "tf"]:
            btss_out[0].events[col] = np.zeros((btss[0].events.shape[0],))
        return btss_out




class OptogeneticStimulus(Stimulus):
    """

    """


    def __init__(
            self, 
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
                        )

        # Stack photodiode and vstim dataframes.
        vstim_events = self._map_btss_vstim(self._VStim)
        self.events = pd.concat((self._Photodiode.events, vstim_events), axis=1)
        print(self._Photodiode.events, vstim_events, self.events)


    def _map_btss_vstim(
            self, 
            vstim, 
        ):
        """"""

        # Define stimulus parameters from rig log to keep.
        param_names = ["contrast", "posx", "posy", "ori", "sf", "phase", "tf", "itrial", "istim"]
        vstim_df_mapped = vstim.events[param_names]
        return vstim_df_mapped


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




class SquarePulse(OptogeneticStimulus):
    """

    """


    def __init__(
            self, 
            btss=None, 
            nidaq_ap=None, 
            nidaq_lf=None, 
        ):
        """"""

        btss = self._match_vstim_df(btss)

        super().__init__(
                         btss=btss, 
                         nidaq_ap=nidaq_ap, 
                         nidaq_lf=nidaq_lf, 
                        )

        self._get_events(params=["itrial"])
        self._get_shape(params=["itrial"])
        self._get_unique()


    def _get_unique(
            self, 
        ):
        """"""
        
        self.unique = []
        for itrial in sorted(np.unique(self.events["itrial"])):
                    self.unique.append((itrial))
        return None


    def _match_vstim_df(
            self, 
            btss, 
        ):
        """"""

        btss_out = btss
        for col in ["ori", "sf", "phase", "tf"]:
            btss_out[0].events[col] = np.zeros((btss[0].events.shape[0],))
        return btss_out
