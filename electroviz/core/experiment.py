# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.population import Population
from electroviz.optogeneticstimulus import OptogeneticStimulus

class Experiment:
    '''
    docstring
    '''
    
    def __init__(
            self, 
            Dataset, 
            data_in, 
            data_type, 
        ):
        """"""
        print('Experiment')
        if data_type == "Allen_NWB":
            # Create Population instance
            electrodes_df = data_in.electrodes.to_dataframe()
            # get units dataframe
            units_df = data_in.units.to_dataframe()
            # create population
            self.Population = Population(self, units_df, electrodes_df)
            
            # Create Stimulus instance(s)
            if 'optotagging' in list(data_in.processing.keys()):
                optotagging = data_in.processing['optotagging'].data_interfaces['optogenetic_stimulation']
                self.Stimulus = OptogeneticStimulus(optotagging)
        