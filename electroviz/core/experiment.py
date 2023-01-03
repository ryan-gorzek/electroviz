# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import *
from electroviz.streams.nidaq import NIDAQ
from electroviz.streams.imec import Imec
from electroviz.streams.kilosort import Kilosort
from electroviz.streams.btss import bTsS
from electroviz.core.stimulus import SparseNoise

class Experiment:
    """

    """

    #### Change lists into dicts
    
    def __init__(
            self, 
            experiment_path, 
            SGLX_name="ephys", 
            bTsS_names=["ipsi_random_squares"], 
        ):
        """"""

        # Parse the specified path to experiment directory.
        SGLX_dir, bTsS_dir = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
        # 
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        self.nidaq = NIDAQ(nidaq_dir)
        self.imec = Imec(imec_dir)
        total_imec_samples = self.imec[0].total_samples
        self.kilosort = Kilosort(imec_dir, total_imec_samples)
        # # Align the NIDAQ and Imec syncs.
        # # nidaq_drop, imec_drop = align_sync(self.nidaq, self.imec)
        # Parse bTsS directory.
        btss_dir = experiment_path + bTsS_dir
        self.btss = bTsS(btss_dir)
        # Create Stimulus objects.
        self.stimuli = []
        self.stimuli.append(SparseNoise(self.nidaq, self.btss))
        # Create Population object.
        


    # def __repr__(
    #         self, 
    #     ):
    #     """"""
        