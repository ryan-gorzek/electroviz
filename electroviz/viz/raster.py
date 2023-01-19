
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from matplotlib import use as mpl_use
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1
from scipy.stats import zscore

class Raster:
    """

    """


    def __new__(
            self, 
            time_window, 
            responses, 
            z_score=True, 
            ylabel="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        aspect_ratio = 1.75 / (responses.shape[0] / responses.shape[1])
        if z_score == True:
            ax.imshow(zscore(responses, axis=1), 
                      cmap="binary", 
                      aspect=aspect_ratio)
        else:
            ax.imshow(responses, 
                      cmap="binary", 
                      aspect=aspect_ratio)
        ax.set_xticks(np.linspace(0, responses.shape[1], 6))
        ax.set_xticklabels(np.linspace(*time_window, 6))
        ax.set_xlabel("Time from Onset (ms)", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.show(block=False)
        plt.tight_layout()
        fig.set_size_inches(6, 9)

