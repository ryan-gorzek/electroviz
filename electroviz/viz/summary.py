
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from itertools import chain
from electroviz.utils.cross_corr import cross_corr

class UnitSummary:
    """

    """


    def __new__(
            self, 
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
        ax_kern_isg.set_xticklabels(xticklabels, rotation=45)

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

        fig.suptitle("Unit #" + str(unit.ID))
        plt.show(block=False)
        fig.set_size_inches(30, 15)




class PairSummary:
    """

    """


    def __new__(
            self, 
            stimuli, 
            units, 
            kernels, 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig = plt.figure()

        #### Unit 0

        # Unit raster.
        gs_1 = fig.add_gridspec(12, 6, hspace=1, wspace=1, left=0.03, right=0.23, top=0.93, bottom=0.08)
        ax_raster_0 = fig.add_subplot(gs_1[:13, :3])
        units[0].plot_raster(list(chain(*stimuli)), ax_in=ax_raster_0)

        # Stimulus PSTHs.
        ax_psth_csn_0 = fig.add_subplot(gs_1[:3, 3:6])
        units[0].plot_PSTH(stimuli[0], ax_in=ax_psth_csn_0)
        ax_psth_csn_0.set_xticklabels([])
        ax_psth_csn_0.set_xlabel("")
        ax_psth_csn_0.set_ylabel("")
        ax_psth_csn_0.set_title("Spike Rate (Hz)")

        ax_psth_isn_0 = fig.add_subplot(gs_1[3:6, 3:6])
        units[0].plot_PSTH(stimuli[1], ax_in=ax_psth_isn_0)
        ax_psth_isn_0.set_xticklabels([])
        ax_psth_isn_0.set_xlabel("")
        ax_psth_isn_0.set_ylabel("")

        ax_psth_csg_0 = fig.add_subplot(gs_1[6:9, 3:6])
        units[0].plot_PSTH(stimuli[2], ax_in=ax_psth_csg_0)
        ax_psth_csg_0.set_xticklabels([])
        ax_psth_csg_0.set_xlabel("")
        ax_psth_csg_0.set_ylabel("")

        ax_psth_isg_0 = fig.add_subplot(gs_1[9:13, 3:6])
        units[0].plot_PSTH(stimuli[3], ax_in=ax_psth_isg_0)
        ax_psth_isg_0.set_ylabel("")

        # Contra Sparse Noise Kernels.
        gs_2 = fig.add_gridspec(3, 3, hspace=0.01, wspace=1, left=0.26, right=0.36, top=0.95, bottom=0.73)
        ax_kon_csn_0 = fig.add_subplot(gs_2[:1, :3])
        ax_koff_csn_0 = fig.add_subplot(gs_2[1:2, :3])
        ax_kdiff_csn_0 = fig.add_subplot(gs_2[2:3, :3])
        kernels[0][0].plot_raw(ax_in=np.array((ax_kon_csn_0, ax_koff_csn_0, ax_kdiff_csn_0)))
        ax_kon_csn_0.set_title("")
        ax_koff_csn_0.set_title("")
        ax_kdiff_csn_0.set_title("")

        # Ipsi Sparse Noise Kernels.
        gs_3 = fig.add_gridspec(3, 3, hspace=0.01, wspace=1, left=0.26, right=0.36, top=0.72, bottom=0.5)
        ax_kon_isn_0 = fig.add_subplot(gs_3[:1, :3])
        ax_koff_isn_0 = fig.add_subplot(gs_3[1:2, :3])
        ax_kdiff_isn_0 = fig.add_subplot(gs_3[2:3, :3])
        kernels[0][1].plot_raw(ax_in=np.array((ax_kon_isn_0, ax_koff_isn_0, ax_kdiff_isn_0)))
        ax_kon_isn_0.set_title("")
        ax_koff_isn_0.set_title("")
        ax_kdiff_isn_0.set_title("")

        # Orisf Kernels.
        gs_4 = fig.add_gridspec(6, 3, hspace=1, wspace=1, left=0.26, right=0.36, top=0.48, bottom=0.08)
        ax_kern_csg_0 = fig.add_subplot(gs_4[:3, :3])
        kernels[0][2].plot_raw(ax_in=ax_kern_csg_0)
        ax_kern_csg_0.set_xticklabels([])
        ax_kern_csg_0.set_xlabel("")

        ax_kern_isg_0 = fig.add_subplot(gs_4[3:6, :3])
        kernels[0][3].plot_raw(ax_in=ax_kern_isg_0)
        xticklabels = ax_kern_isg_0.get_xticklabels()
        ax_kern_isg_0.set_xticklabels(xticklabels, rotation=45)

        # Cross-Correlations.
        xcorr_raw, xcorr_filt = cross_corr(units)
        gs_5 = fig.add_gridspec(12, 6, hspace=1, wspace=1, left=0.39, right=0.59, top=0.93, bottom=0.08)
        ax_xcorr_raw = fig.add_subplot(gs_5[:6, :6])
        ax_xcorr_raw.bar(range(-50, 51), xcorr_raw, color=(0.9, 0.2, 0.2))
        ax_xcorr_raw.set_xlabel("Time from Onset (ms)")
        ax_xcorr_filt = fig.add_subplot(gs_5[6:13, :6])
        ax_xcorr_filt.hlines(0, -50, 50, colors="k", linestyles="--")
        ax_xcorr_filt.plot(range(-50, 51), xcorr_filt, color=(0.9, 0.2, 0.2))
        ax_xcorr_filt.set_xlabel("Time from Onset (ms)")
        ylim = ax_xcorr_filt.get_ylim()
        ax_xcorr_filt.fill([2, 2, 5, 5], ylim + ylim[::-1], color=(0.6, 0.6, 0.6, 0.5))
        
        #### Unit 1 (left to right)

        # Contra Sparse Noise Kernels.
        gs_6 = fig.add_gridspec(3, 3, hspace=0.01, wspace=1, left=0.62, right=0.72, top=0.95, bottom=0.73)
        ax_kon_csn_1 = fig.add_subplot(gs_6[:1, :3])
        ax_koff_csn_1 = fig.add_subplot(gs_6[1:2, :3])
        ax_kdiff_csn_1 = fig.add_subplot(gs_6[2:3, :3])
        kernels[1][0].plot_raw(ax_in=np.array((ax_kon_csn_1, ax_koff_csn_1, ax_kdiff_csn_1)))
        ax_kon_csn_1.set_title("")
        ax_koff_csn_1.set_title("")
        ax_kdiff_csn_1.set_title("")

        # Ipsi Sparse Noise Kernels.
        gs_7 = fig.add_gridspec(3, 3, hspace=0.01, wspace=1, left=0.62, right=0.72, top=0.72, bottom=0.5)
        ax_kon_isn_1 = fig.add_subplot(gs_7[:1, :3])
        ax_koff_isn_1 = fig.add_subplot(gs_7[1:2, :3])
        ax_kdiff_isn_1 = fig.add_subplot(gs_7[2:3, :3])
        kernels[1][1].plot_raw(ax_in=np.array((ax_kon_isn_1, ax_koff_isn_1, ax_kdiff_isn_1)))
        ax_kon_isn_1.set_title("")
        ax_koff_isn_1.set_title("")
        ax_kdiff_isn_1.set_title("")

        # Orisf Kernels.
        gs_8 = fig.add_gridspec(6, 3, hspace=1, wspace=1, left=0.62, right=0.72, top=0.48, bottom=0.08)
        ax_kern_csg_1 = fig.add_subplot(gs_8[:3, :3])
        kernels[1][2].plot_raw(ax_in=ax_kern_csg_1)
        ax_kern_csg_1.set_xticklabels([])
        ax_kern_csg_1.set_xlabel("")

        ax_kern_isg_1 = fig.add_subplot(gs_8[3:6, :3])
        kernels[1][3].plot_raw(ax_in=ax_kern_isg_1)
        xticklabels = ax_kern_isg_1.get_xticklabels()
        ax_kern_isg_1.set_xticklabels(xticklabels, rotation=45)

        # Unit raster.
        gs_9 = fig.add_gridspec(12, 6, hspace=1, wspace=1, left=0.75, right=0.95, top=0.93, bottom=0.08)
        ax_raster_1 = fig.add_subplot(gs_9[:13, 3:6])
        units[1].plot_raster(list(chain(*stimuli)), ax_in=ax_raster_1)
        ax_raster_1.set_ylabel("")
        ax_raster_1.set_yticklabels([])

        # Stimulus PSTHs.
        ax_psth_csn_1 = fig.add_subplot(gs_9[:3, :3])
        units[1].plot_PSTH(stimuli[0], ax_in=ax_psth_csn_1)
        ax_psth_csn_1.set_xticklabels([])
        ax_psth_csn_1.set_xlabel("")
        ax_psth_csn_1.set_ylabel("")
        ax_psth_csn_1.set_title("Spike Rate (Hz)")

        ax_psth_isn_1 = fig.add_subplot(gs_9[3:6, :3])
        units[1].plot_PSTH(stimuli[1], ax_in=ax_psth_isn_1)
        ax_psth_isn_1.set_xticklabels([])
        ax_psth_isn_1.set_xlabel("")
        ax_psth_isn_1.set_ylabel("")

        ax_psth_csg_1 = fig.add_subplot(gs_9[6:9, :3])
        units[1].plot_PSTH(stimuli[2], ax_in=ax_psth_csg_1)
        ax_psth_csg_1.set_xticklabels([])
        ax_psth_csg_1.set_xlabel("")
        ax_psth_csg_1.set_ylabel("")

        ax_psth_isg_1 = fig.add_subplot(gs_9[9:13, :3])
        units[1].plot_PSTH(stimuli[3], ax_in=ax_psth_isg_1)
        ax_psth_isg_1.set_ylabel("")

        plt.show(block=False)
        fig.set_size_inches(40, 20)

        if save_path != "":
            fig.savefig(save_path, bbox_inches="tight")

