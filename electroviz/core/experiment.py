
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import *
from electroviz.streams.nidaq import NIDAQ
from electroviz.streams.imec import ImecAP, ImecLF
from electroviz.streams.kilosort import Kilosort
from electroviz.utils.align_sync import align_ap_sync, align_lf_sync
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
        # Read electroviz config file from experiment directory.
        SGLX_name, bTsS_names = read_config(experiment_path)
        # Parse the specified path to experiment directory.
        SGLX_dir, bTsS_dirs = parse_experiment_dir(experiment_path, SGLX_name, bTsS_names)
        # 
        print("Loading NIDAQ data...")
        nidaq_dir, imec_dir = parse_SGLX_dir(experiment_path + SGLX_dir)
        if not "opto_tagging_pulse" in bTsS_names:
            nidaq = NIDAQ(nidaq_dir, opto=False)
        else:
            nidaq = NIDAQ(nidaq_dir)
        print("Loading Imec data...")
        imec_ap = ImecAP(imec_dir)
        imec_lf = ImecLF(imec_dir)
        total_ap_samples = [im.total_samples for im in imec_ap]
        print("Loading Kilosort data...")
        kilosort = Kilosort(imec_dir, total_ap_samples)
        print("Aligning Syncs...")
        # Align the NIDAQ and Imec syncs.
        self.nidaq_ap, self.imec_ap, self.kilosort = align_ap_sync(nidaq, imec_ap, kilosort)
        self._nidaq = nidaq
        self._imec_lf = imec_lf
        self.nidaq_lf, self.imec_lf, self.drops = align_lf_sync(nidaq, imec_lf)

        del nidaq

        # print("Loading bTsS data...")
        # # Parse bTsS directory.
        # self.btss = []
        # for idx, cdir in enumerate(bTsS_dirs):
        #     btss_dir = experiment_path + cdir
        #     self.btss.append(bTsS(btss_dir, index=idx))
        # print("Creating Stimuli...")
        # # Create Stimulus objects.
        # stimuli = []
        # for btss_name, btss_obj in zip(bTsS_names, self.btss):
        #     if "random_squares" in btss_name:
        #         stimuli.append(SparseNoise(btss_obj, self.nidaq_ap, self.nidaq_lf))
        #     elif "random_gratings" in btss_name:
        #         stimuli.append(StaticGratings(btss_obj, self.nidaq_ap, self.nidaq_lf))
        #     elif "contrast_reversal" in btss_name:
        #         stimuli.append(ContrastReversal(btss_obj, self.nidaq_ap, self.nidaq_lf))
        #     elif "opto_tagging" in btss_name:
        #         stimuli.append(SquarePulse(btss_obj, self.nidaq_ap, self.nidaq_lf))
        # # Reorder Stimulus objects for convenience.
        # stim_order = ["contra_random_squares", "ipsi_random_squares", 
        #             "contra_random_gratings", "ipsi_random_gratings", 
        #             "contrast_reversal", 
        #             "opto_tagging_pulse"]
        # self.stimuli = []
        # for stim in stim_order:
        #     for name, obj in zip(bTsS_names, stimuli):
        #         if stim in name:
        #             self.stimuli.append(obj)
        # print("Creating Populations...")
        # # Create Population object.
        # self.populations = []
        # for im, ks in zip(self.imec_ap, self.kilosort):
        #     self.populations.append(Population(im, ks))
        # print("Creating Probes...")
        # # Create Probe object.
        # self.probes = []
        # for sy, im in np.array(self.imec_lf).reshape(-1, 2):
        #     self.probes.append(Probe(sy, im))

        print("Done!")

