# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import *
from electroviz.streams.nidaq import NIDAQ
from electroviz.streams.imec import Imec

class Experiment:
    """

    """
    
    def __init__(
            self, 
            experiment_path, 
            SGLX_name="ephys", 
            bTsS_names=["ipsi_random_squares"], 
        ):
        """"""

        # Parse the specified path to experiment directory.
        SGLX_dir, bTsS_dir = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
        # Parse SpikeGLX directory.
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        self.nidaq = NIDAQ(nidaq_dir)
        self.imec = Imec(imec_dir)
        # self.kilosort = Kilosort(imec_dir)
        # # Align the NIDAQ and Imec syncs.
        # # nidaq_drop, imec_drop = align_sync(self.nidaq, self.imec)
        # # Parse bTsS directory.
        # btss_dir = path + bTsS_dir
        # self.btss = bTsS(btss_dir)
        