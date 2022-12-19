# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.core.unit import Unit

class Population:
    '''
    docstring
    '''
    
    #### Special Methods ####

    def __init__(
            self, 
            imec_spikes, 
            imec_sync, 
        ):
        """"""

        self.sync = imec_sync
        self.total_units = imec_spikes.total_units
        self.spike_times = imec_spikes.spike_times
        # Create Unit objects.
        self._Units = []
        for uid in range(self.total_units):
            unit = Unit(imec_spikes, imec_sync, uid)
            self._Units.append(unit)
        
        # Define current index for iteration.
        self._current_unit_idx = 0
        

    def __getitem__(
            self, 
            unit_idx, 
        ):
        """"""

        unit = self._Units[unit_idx]
        return unit

    def __iter__(self):
        return self

    def __next__(self):
        """"""
        if self._current_unit_idx < self.total_units:
            unit = self._Units[self._current_unit_idx]
            self._current_unit_idx += 1
            return unit

    
    #### Plotting Methods ####

    
    
    #### Quantification Methods ####
    
    
    
    #### Data Management Methods ####
    
    
    
    #### Private Methods ####
    
    
        