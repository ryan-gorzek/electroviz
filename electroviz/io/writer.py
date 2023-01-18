# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

def write_for_matlab(
        population, 
        stimulus, 
        save_path="", 
    ):
    """"""

    dict_out = {}

    dict_out["spike_clusters"] = population.spike_times.tocoo().row
    dict_out["spike_times"] = population.spike_times.tocoo().col
    dict_out["spikes_matrix_shape"] = np.array(population.spike_times.shape)
    dict_out["unit_id"] = np.array(population.units["unit_id"])
    dict_out["peak_channel"] = np.array(population.units["depth"])
    dict_out["spike_rate"] = np.array(population.units["spike_rate"])
    dict_out["sample_onset"] = np.array(stimulus.events["sample_onset"])
    dict_out["contrast"] = np.array(stimulus.events["contrast"])
    dict_out["posx"] = np.array(stimulus.events["posx"])
    dict_out["posy"] = np.array(stimulus.events["posy"])
    dict_out["ori"] = np.array(stimulus.events["ori"])
    dict_out["sf"] = np.array(stimulus.events["sf"])
    dict_out["phase"] = np.array(stimulus.events["phase"])
    dict_out["itrial"] = np.array(stimulus.events["itrial"])

    for name, arr in dict_out.items():
        path = save_path + name + ".npy"
        np.save(path, arr)
