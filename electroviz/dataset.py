# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
from pynwb import NWBHDF5IO
from electroviz.experiment import Experiment

class Dataset:
    '''
    docstring
    '''
    
    def __init__(self, nwb_file_path, experiment_name="default"):
        print('Dataset')
        if os.path.isfile(nwb_file_path) == True:
            nwb_io = NWBHDF5IO(nwb_file_path, mode='r', load_namespaces=True)
            nwb_file = nwb_io.read()
            # add metadata
            if experiment_name == "default":
                exp_name = "Experiment"
            setattr(self, exp_name, Experiment(self, nwb_file, experiment_name=exp_name))
        else:
            raise Exception('Cannot load the specified file.')
        
        