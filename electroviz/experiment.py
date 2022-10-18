# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Experiment:
    '''
    docstring
    '''
    
    def __init__(self, source_data):
        print('Experiment')
        from electroviz.population import Population
        #
        electrodes_df = source_data.electrodes.to_dataframe()
        # get units dataframe
        units_df = source_data.units.to_dataframe()
        # create population
        self.Population = Population(units_df, electrodes_df)
        