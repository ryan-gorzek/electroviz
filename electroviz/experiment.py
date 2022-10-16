# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Experiment:
    '''
    docstring
    '''
    
    def __init__(self, SourceData):
        print('Experiment')
        from electroviz.unit import Unit
        from electroviz.population import Population
        # get units and create population
        unit_df = SourceData.units.to_dataframe()
        unit_ids = unit_df.index.values
        all_units = []
        for uid in unit_ids:
            all_units.append(Unit(unit_df.loc[uid]))
        self.Population = Population(all_units)
        