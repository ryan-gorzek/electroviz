# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

def plot_sparse_noise_response(
        unit, 
        stimulus, 
        time_window=[-0.02, 0.100], 
    ):
    """"""

    sample_window = np.array(time_window)*30000
    num_samples = int(sample_window[1] - sample_window[0])
    responses = np.zeros((int(num_samples/30), 14, 10, 2, 5))
    for event in stimulus:
        window = (sample_window + event.sample_onset).astype(int)
        resp = unit.get_spike_times(window)
        x_idx, y_idx, c_idx = stimulus.get_params_index((event.posx, event.posy, event.contrast))
        responses[:, x_idx, y_idx, c_idx, int(event.itrial)] = np.sum(resp.reshape(120, -1), axis=1)
    
    mpl_use("Qt5Agg")
    fig, axs = plt.subplots(10, 14)
    for posx, posy, contrast in stimulus.unique:
        x_idx, y_idx, c_idx = stimulus.get_params_index((posx, posy, contrast))
        mean_response = np.nanmean(responses[:, x_idx, y_idx, c_idx, :].squeeze(), axis=1)
        if c_idx == 0:
            axs[y_idx, x_idx].plot(mean_response, color="b")
        else:
            axs[y_idx, x_idx].plot(mean_response, color="r")
        # axs[y_idx, x_idx].axis("off")
        axs[y_idx, x_idx].axvline(20, color="k")
    plt.show()

