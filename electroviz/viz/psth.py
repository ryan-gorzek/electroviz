# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from matplotlib import use as mpl_use
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1

class PSTH:
    """

    """


    def __new__(
            self, 
            time_window, 
            responses, 
        ):
        """"""

        colors = ((0.0, 0.0, 0.0), (0.9, 0.2, 0.2))
        mpl_use("Qt5Agg")
        fig, ax = plt.subplots()
        
        if responses.ndim == 2:
            for resp, col in zip(responses, colors[:responses.ndim]):
                t = np.linspace(0, resp.size, resp.size)
                ax.bar(t, resp, color=col)
        else:
            t = np.linspace(0, responses.size, responses.size)
            ax.bar(t, responses, color="k")
        ax.set_xticks(np.linspace(0, t.size, 6))
        ax.set_xticklabels(np.linspace(*time_window, 6))
        ax.set_xlabel("Time from Onset (ms)", fontsize=16)
        ends = np.array((responses.min(), responses.max()))
        rng = np.diff(ends)
        lims = ends + 0.1 * np.array((-rng, rng)).T
        ax.set_ylim(lims[0])
        ax.set_ylabel("Spikes / second", fontsize=16)
        plt.show(block=False)
        fig.set_size_inches(6, 6)
