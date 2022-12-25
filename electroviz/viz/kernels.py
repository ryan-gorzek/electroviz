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
    num_samples = int(sample_window[1] - sample_window[0] + 1)
    responses = np.zeros((num_samples, 14, 10, 2))
    for (onset, offset), (x_idx, y_idx, contrast_idx) in stimulus:
        window = (sample_window + onset).astype(int)
        resp = unit.get_spike_times(window)
        responses[:, x_idx, y_idx, contrast_idx] += resp
    
    mpl_use("Qt5Agg")
    fig, axs = plt.subplots(10, 14)
    for _, (x_idx, y_idx, contrast_idx) in stimulus:
        if contrast_idx == 0:
            axs[y_idx, x_idx].plot(responses[:, x_idx, y_idx, contrast_idx], color="b")
        else:
            axs[y_idx, x_idx].plot(responses[:, x_idx, y_idx, contrast_idx], color="r")
        axs[y_idx, x_idx].axis("off")
    plt.show()

