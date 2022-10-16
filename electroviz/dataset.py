# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Dataset:
    '''
    docstring
    '''
    
    def __init__(self, SourceDataPath):
        print('Dataset')
        from electroviz.experiment import Experiment
        import os
        from pynwb import NWBHDF5IO
        if os.path.isfile(SourceDataPath) == True:
            NWBIO = NWBHDF5IO(SourceDataPath, mode='r', load_namespaces=True)
            NWBFile = NWBIO.read()
            # add metadata
            self.Experiment = Experiment(NWBFile)
        else:
            raise Exception('Cannot load the specified file.')
        
        