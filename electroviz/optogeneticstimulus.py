# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.stimulus import Stimulus

class OptogeneticStimulus(Stimulus):
    '''
    docstring
    '''
    
    def __init__(self, time_intervals_object):
        print('OptogeneticStimulus')
        super().__init__(time_intervals_object)
