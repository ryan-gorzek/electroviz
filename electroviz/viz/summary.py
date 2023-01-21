
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from itertools import chain

class UnitSummary:
    """

    """


        def __new__(
                unit, 
                stimuli, 
                kernels, 
            ):
            """"""

            mpl_use("Qt5Agg")
            fig = plt.figure()

            gs_1 = fig.add_gridspec(12, 6, hspace=1, wspace=1, left=0.04, right=0.36, top=0.93, bottom=0.08)
            ax_raster = fig.add_subplot(gs_1[:9, :3])
            unit._Population.plot_raster(None, responses=unit._Population._responses, ax_in=ax_raster)
            ax_raster.set_xticklabels([])
            ax_raster.set_xlabel("")
            ax_raster.set_title("Population Responses")
            ax_xcorr = fig.add_subplot(gs_1[:9, 3:6])
            # ax_xcorr.axis("off")
            ax_psth = fig.add_subplot(gs_1[9:13, :3])
            unit.plot_PSTH(list(chain(*stimuli)), ax_in=ax_psth)
            ax_peaks = fig.add_subplot(gs_1[9:13, 3:6])
            # ax_peaks.axis("off")

            # Stimulus PSTHs.
            gs_2 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.40, right=0.56, top=0.93, bottom=0.08)
            ax_psth_csn = fig.add_subplot(gs_2[:3, :3])
            unit.plot_PSTH(stimuli[0], ax_in=ax_psth_csn)
            ax_psth_csn.set_xticklabels([])
            ax_psth_csn.set_xlabel("")
            ax_psth_csn.set_ylabel("")
            ax_psth_csn.set_title("Spike Rate (Hz)")

            ax_psth_isn = fig.add_subplot(gs_2[3:6, :3])
            unit.plot_PSTH(stimuli[1], ax_in=ax_psth_isn)
            ax_psth_isn.set_xticklabels([])
            ax_psth_isn.set_xlabel("")
            ax_psth_isn.set_ylabel("")

            ax_psth_csg = fig.add_subplot(gs_2[6:9, :3])
            unit.plot_PSTH(stimuli[2], ax_in=ax_psth_csg)
            ax_psth_csg.set_xticklabels([])
            ax_psth_csg.set_xlabel("")
            ax_psth_csg.set_ylabel("")

            ax_psth_isg = fig.add_subplot(gs_2[9:13, :3])
            unit.plot_PSTH(stimuli[3], ax_in=ax_psth_isg)
            ax_psth_isg.set_ylabel("")

            # Contra Sparse Noise Kernels.
            gs_3 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.60, right=0.76, top=0.96, bottom=0.73)
            ax_kon_csn = fig.add_subplot(gs_3[:1, :3])
            ax_koff_csn = fig.add_subplot(gs_3[1:2, :3])
            ax_kdiff_csn = fig.add_subplot(gs_3[2:3, :3])
            kernels[0].plot_raw(ax_in=np.array((ax_kon_csn, ax_koff_csn, ax_kdiff_csn)))
            ax_kon_csn.set_title("")
            ax_koff_csn.set_title("")
            ax_kdiff_csn.set_title("")

            # Ipsi Sparse Noise Kernels.
            gs_4 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.60, right=0.76, top=0.72, bottom=0.5)
            ax_kon_isn = fig.add_subplot(gs_4[:1, :3])
            ax_koff_isn = fig.add_subplot(gs_4[1:2, :3])
            ax_kdiff_isn = fig.add_subplot(gs_4[2:3, :3])
            kernels[1].plot_raw(ax_in=np.array((ax_kon_isn, ax_koff_isn, ax_kdiff_isn)))
            ax_kon_isn.set_title("")
            ax_koff_isn.set_title("")
            ax_kdiff_isn.set_title("")

            # Orisf Kernels.
            gs_5 = fig.add_gridspec(6, 6, hspace=1, wspace=1, left=0.60, right=0.76, top=0.48, bottom=0.08)
            ax_kern_csg = fig.add_subplot(gs_5[:3, :3])
            kernels[2].plot_raw(ax_in=ax_kern_csg)
            ax_kern_csg.set_xticklabels([])
            ax_kern_csg.set_xlabel("")

            ax_kern_isg = fig.add_subplot(gs_5[3:6, :3])
            kernels[3].plot_raw(ax_in=ax_kern_isg)
            xticklabels = ax_kern_isg.get_xticklabels()
            ax_kern_isg.set_xticklabels(xticklabels, rotation = 45)

            # Norms.
            gs_5 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.80, right=0.96, top=0.93, bottom=0.08)
            ax_norm_csn = fig.add_subplot(gs_5[:3, :3])
            kernels[0].plot_norm_delay(ax_in=ax_norm_csn)
            ax_norm_csn.set_xticklabels([])
            ax_norm_csn.set_xlabel("")
            ax_norm_csn.set_ylabel("")
            ax_norm_csn.set_title(f"||Kernel||\N{SUPERSCRIPT TWO}")

            ax_norm_isn = fig.add_subplot(gs_5[3:6, :3])
            kernels[1].plot_norm_delay(ax_in=ax_norm_isn)
            ax_norm_isn.set_xticklabels([])
            ax_norm_isn.set_xlabel("")
            ax_norm_isn.set_ylabel("")

            ax_norm_csg = fig.add_subplot(gs_5[6:9, :3])
            kernels[2].plot_norm_delay(ax_in=ax_norm_csg)
            ax_norm_csg.set_xticklabels([])
            ax_norm_csg.set_xlabel("")
            ax_norm_csg.set_ylabel("")

            ax_norm_isg = fig.add_subplot(gs_5[9:13, :3])
            kernels[3].plot_norm_delay(ax_in=ax_norm_isg)
            ax_norm_isg.set_ylabel("")

            fig.suptitle("Unit #" + str(self.ID))
            plt.show(block=False)
            fig.set_size_inches(30, 15)
