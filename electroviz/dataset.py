# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Dataset:
    '''
    docstring
    '''
    
    def __init__(self, nwb_file_path):
        print('Dataset')
        from electroviz.experiment import Experiment
        import os
        from pynwb import NWBHDF5IO
        if os.path.isfile(nwb_file_path) == True:
            nwb_io = NWBHDF5IO(nwb_file_path, mode='r', load_namespaces=True)
            nwb_file = nwb_io.read()
            # add metadata
            self.Experiment = Experiment(nwb_file)
        else:
            raise Exception('Cannot load the specified file.')
        
        