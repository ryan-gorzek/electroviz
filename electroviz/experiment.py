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
    
    def __init__(self, nwb_file):
        print('Experiment')
        # Create Population instance
        electrodes_df = nwb_file.electrodes.to_dataframe()
        # get units dataframe
        units_df = nwb_file.units.to_dataframe()
        # create population
        self.Population = Population(units_df, electrodes_df)
        
        # Create Stimulus instance(s)
        if 'optotagging' in list(nwb_file.processing.keys()):
            optotagging = nwb_file.processing['optotagging'].data_interfaces['optogenetic_stimulation']
            self.OptogeneticStimulus = OptogeneticStimulus(optotagging)
        