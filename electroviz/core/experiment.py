
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import *
from electroviz.streams.nidaq import NIDAQ
from electroviz.streams.imec import ImecAP, ImecLF
from electroviz.streams.kilosort import Kilosort
from electroviz.utils.align_sync import align_sync
from electroviz.streams.btss import bTsS
from electroviz.core.stimulus import SparseNoise, StaticGratings, ContrastReversal, SquarePulse
from electroviz.core.population import Population
from electroviz.core.Probe import Probe

class Experiment:
    """

    """


    def __init__(
            self, 
            experiment_path, 
        ):
        """"""

        print("Parsing directories...")
        SGLX_name, NIDAQ_gates, bTsS_names = read_config(experiment_path)
        SGLX_dir, bTsS_dirs = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
    
        print("Loading NIDAQ data...")
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        nidaq = NIDAQ(nidaq_dir, NIDAQ_gates)

        print("Loading bTsS data...")
        self.btss = []
        for idx, cdir in enumerate(bTsS_dirs):
            btss_dir = experiment_path + cdir
            self.btss.append(bTsS(btss_dir, index=idx))

        print("Loading Imec data...")
        self.imec_ap = ImecAP(imec_dir)
        self.imec_lf = ImecLF(imec_dir)

        print("Loading Kilosort data...")
        total_ap_samples = [im.total_samples for im in self.imec_ap]
        self.kilosort = Kilosort(imec_dir, total_ap_samples)

        print("Aligning Syncs...")
        self.nidaq_ap = align_sync(nidaq, self.imec_ap[0], nidaq_dir, type="AP")
        self.nidaq_lf = align_sync(nidaq, self.imec_lf[0], nidaq_dir, type="LF")

        print("Creating Stimuli...")
        stimuli = []
        for nidaq_ap, nidaq_lf, btss, name in zip(self.nidaq_ap, self.nidaq_lf, self.btss, bTsS_names):
            if "random_squares" in name:
                stimuli.append(SparseNoise(btss, nidaq_ap, nidaq_lf))
            elif "random_gratings" in name:
                stimuli.append(StaticGratings(btss, nidaq_ap, nidaq_lf))
            elif "contrast_reversal" in name:
                stimuli.append(ContrastReversal(btss, nidaq_ap, nidaq_lf))
            elif "opto_tagging" in name:
                stimuli.append(SquarePulse(btss, nidaq_ap, nidaq_lf))
        # Reorder Stimulus objects for convenience.
        stim_order = ["contra_random_squares", "ipsi_random_squares", 
                      "contra_random_gratings", "ipsi_random_gratings", 
                      "contrast_reversal", 
                      "opto_tagging_pulse"]
        self.stimuli = []
        for stim in stim_order:
            for name, obj in zip(bTsS_names, stimuli):
                if stim in name:
                    self.stimuli.append(obj)
        
        print("Creating Populations...")
        # Create Population object.
        self.populations = []
        for im, ks in zip(self.imec_ap, self.kilosort):
            self.populations.append(Population(im, ks))
        
        print("Creating Probes...")
        # Create Probe object.
        self.probes = []
        for sy, im in np.array(self.imec_lf).reshape(-1, 2):
            self.probes.append(Probe(im, sy))

        print("Done!")

