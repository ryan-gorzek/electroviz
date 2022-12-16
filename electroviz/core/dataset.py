# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
from electroviz.experiment import Experiment

class Dataset:
    '''
    docstring
    '''
    
    def __init__(
            self, 
            data_path, 
            data_type, 
        ):
        valid_types = ["Allen_NWB"]
        if data_type not in valid_types:
            raise ValueError("Invalid data type. Data type must be one of: {}".format(*valid_types))
        print('Dataset')
        if (data_type == "Allen_NWB") & (os.path.isfile(data_path) == True):
            from pynwb import NWBHDF5IO
            nwb_io = NWBHDF5IO(data_path, mode='r', load_namespaces=True)
            nwb_file = nwb_io.read()
            # add metadata
            self.Experiment = Experiment(self, nwb_file, data_type="Allen_NWB")
        else:
            raise Exception('Cannot load the specified file.')
        
        