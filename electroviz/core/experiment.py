
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
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


    def __init__(
            self, 
            experiment_path, 
        ):
        """"""

        # Read electroviz config file from experiment directory.
        SGLX_name, bTsS_names = read_config(experiment_path)
        # Parse the specified path to experiment directory.
        SGLX_dir, bTsS_dirs = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
        # 
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        nidaq = NIDAQ(nidaq_dir)
        imec = Imec(imec_dir)
        total_imec_samples = [im.total_samples for im in imec]
        kilosort = Kilosort(imec_dir, total_imec_samples)
        # Align the NIDAQ and Imec syncs.
        self.nidaq, self.imec, self.kilosort = align_sync(nidaq, imec, kilosort)
        # Parse bTsS directory.
        self.btss = []
        for idx, cdir in enumerate(bTsS_dirs):
            btss_dir = experiment_path + cdir
            self.btss.append(bTsS(btss_dir, index=idx))
        # Create Stimulus objects.
        self.stimuli = []
        for btss_name, btss_obj in zip(bTsS_names, self.btss):
            if "random_squares" in btss_name:
                self.stimuli.append(SparseNoise(self.nidaq, btss_obj))
            elif "random_gratings" in btss_name:
                self.stimuli.append(StaticGratings(self.nidaq, btss_obj))
            # elif "contrast_reversal" in name:
            #     self.stimuli.append(ContrastReversal(self.nidaq, self.btss))
            # elif "opto_tagging" in name:
            #     self.stimuli.append(SquarePulse(self.nidaq, self.btss))
        # Create Population object.
        self.populations = []
        for im, ks in zip(self.imec, self.kilosort):
            self.populations.append(Population(im, ks))

