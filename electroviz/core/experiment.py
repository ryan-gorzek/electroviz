# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import *
from electroviz.streams.nidaq import NIDAQ
from electroviz.streams.imec import Imec
from electroviz.streams.kilosort import Kilosort
from electroviz.utils.align_sync import align_sync
from electroviz.streams.btss import bTsS
from electroviz.core.stimulus import SparseNoise, StaticGratings
from electroviz.core.population import Population

class Experiment:
    """

    """

    #### Change lists into dicts
    
    def __init__(
            self, 
            experiment_path, 
            SGLX_name="ephys", 
            bTsS_names=["contra_random_squares"], # ALSO CHANGE STIM TYPE BELOW
        ):
        """"""

        # Parse the specified path to experiment directory.
        SGLX_dir, bTsS_dir = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
        # 
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        nidaq = NIDAQ(nidaq_dir)
        imec = Imec(imec_dir)
        total_imec_samples = [im.total_samples for im in imec]
        kilosort = Kilosort(imec_dir, total_imec_samples)
        # Align the NIDAQ and Imec syncs.
        self.nidaq, self.imec, self.kilosort = align_sync(nidaq, imec, kilosort)
        # Parse bTsS directory.
        btss_dir = experiment_path + bTsS_dir
        self.btss = bTsS(btss_dir)
        # Create Stimulus objects.
        self.stimuli = []
        for name in bTsS_names:
            if "random_squares" in name:
                self.stimuli.append(SparseNoise(self.nidaq, self.btss))
            elif "random_gratings" in name:
                self.stimuli.append(StaticGratings(self.nidaq, self.btss))
        # Create Population object.
        self.populations = []
        for im, ks in zip(self.imec, self.kilosort):
            self.populations.append(Population(im, ks))


    # def __repr__(
    #         self, 
    #     ):
    #     """"""
        