
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig = plt.figure()

        # Unit Raster.
        gs_1 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.04, right=0.20, top=0.93, bottom=0.08)
        ax_raster = fig.add_subplot(gs_1[:13, :3])
        unit.plot_raster(list(chain(*stimuli)), ax_in=ax_raster)
        ax_raster.set_title("Stimulus Events")
        ax_raster.set_xlim((-1.25, 53.25))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        prev_len = 0
        for stim, color in zip(stimuli, colors):
            stim_len = len(stim)
            ax_raster.add_patch(Rectangle((51.25, prev_len), 2, stim_len, color=color))
            prev_len += stim_len
        ax_raster.spines["right"].set_visible(False)
        ax_raster.set_yticklabels([])
        ax_raster.set_ylabel("")
        ax_raster.set_yticks([])

        # Stimulus PSTHs.
        gs_2 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.24, right=0.40, top=0.93, bottom=0.08)
        ax_psth_csn = fig.add_subplot(gs_2[:3, :3])
        unit.plot_PSTH(stimuli[0], ax_in=ax_psth_csn)
        ax_psth_csn.set_xticklabels([])
        ax_psth_csn.set_xlabel("")
        ax_psth_csn.set_xlim((-0.75, 51.75))
        ax_psth_csn.set_ylabel("Contra Sparse Noise", color=colors[0])
        ax_psth_csn.set_title("Spike Rate (Hz)")

        ax_psth_isn = fig.add_subplot(gs_2[3:6, :3])
        unit.plot_PSTH(stimuli[1], ax_in=ax_psth_isn)
        ax_psth_isn.set_xticklabels([])
        ax_psth_isn.set_xlabel("")
        ax_psth_isn.set_xlim((-0.75, 51.75))
        ax_psth_isn.set_ylabel("Ipsi Sparse Noise", color=colors[1])

        ax_psth_csg = fig.add_subplot(gs_2[6:9, :3])
        unit.plot_PSTH(stimuli[2], ax_in=ax_psth_csg)
        ax_psth_csg.set_xticklabels([])
        ax_psth_csg.set_xlabel("")
        ax_psth_csg.set_xlim((-0.75, 51.75))
        ax_psth_csg.set_ylabel("Contra Static Gratings", color=colors[2])

        ax_psth_isg = fig.add_subplot(gs_2[9:13, :3])
        unit.plot_PSTH(stimuli[3], ax_in=ax_psth_isg)
        ax_psth_isg.set_xlim((-0.75, 51.75))
        ax_psth_isg.set_ylabel("Ipsi Static Gratings", color=colors[3])

        ylims = []
        for ax in [ax_psth_csn, ax_psth_isn, ax_psth_csg, ax_psth_isg]:
            ylim = ax.get_ylim()
            ylims.append(ylim[1])
        for ax in [ax_psth_csn, ax_psth_isn, ax_psth_csg, ax_psth_isg]:
            ax.set_ylim((0, np.max(ylims)))

        # Contra Sparse Noise Kernels.
        gs_3 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.44, right=0.60, top=0.95, bottom=0.73)
        ax_kon_csn = fig.add_subplot(gs_3[:1, :3])
        ax_koff_csn = fig.add_subplot(gs_3[1:2, :3])
        ax_kdiff_csn = fig.add_subplot(gs_3[2:3, :3])
        kernels[0].plot_raw(ax_in=np.array((ax_kon_csn, ax_koff_csn, ax_kdiff_csn)))
        ax_kon_csn.axis("on")
        ax_kon_csn.set_frame_on(False)
        ax_kon_csn.set_xticks([])
        ax_kon_csn.set_yticks([])
        ax_kon_csn.set_title("Peak")
        ax_kon_csn.set_ylabel("ON")
        ax_koff_csn.axis("on")
        ax_koff_csn.set_frame_on(False)
        ax_koff_csn.set_xticks([])
        ax_koff_csn.set_yticks([])
        ax_koff_csn.set_title("")
        ax_koff_csn.set_ylabel("OFF")
        ax_kdiff_csn.axis("on")
        ax_kdiff_csn.set_frame_on(False)
        ax_kdiff_csn.set_xticks([])
        ax_kdiff_csn.set_yticks([])
        ax_kdiff_csn.set_title("")
        ax_kdiff_csn.set_ylabel("ON - OFF")
        ax_kon_csn_v = fig.add_subplot(gs_3[:1, 3:6])
        ax_koff_csn_v = fig.add_subplot(gs_3[1:2, 3:6])
        ax_kdiff_csn_v = fig.add_subplot(gs_3[2:3, 3:6])
        kernels[0].plot_raw(ax_in=np.array((ax_kon_csn_v, ax_koff_csn_v, ax_kdiff_csn_v)), 
                            type="valley")
        ax_kon_csn_v.set_title("Valley")
        ax_koff_csn_v.set_title("")
        ax_kdiff_csn_v.set_title("")

        # Ipsi Sparse Noise Kernels.
        gs_4 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.44, right=0.60, top=0.72, bottom=0.5)
        ax_kon_isn = fig.add_subplot(gs_4[:1, :3])
        ax_koff_isn = fig.add_subplot(gs_4[1:2, :3])
        ax_kdiff_isn = fig.add_subplot(gs_4[2:3, :3])
        kernels[1].plot_raw(ax_in=np.array((ax_kon_isn, ax_koff_isn, ax_kdiff_isn)))
        ax_kon_isn.axis("on")
        ax_kon_isn.set_frame_on(False)
        ax_kon_isn.set_xticks([])
        ax_kon_isn.set_yticks([])
        ax_kon_isn.set_title("")
        ax_kon_isn.set_ylabel("ON")
        ax_koff_isn.axis("on")
        ax_koff_isn.set_frame_on(False)
        ax_koff_isn.set_xticks([])
        ax_koff_isn.set_yticks([])
        ax_koff_isn.set_title("")
        ax_koff_isn.set_ylabel("OFF")
        ax_kdiff_isn.axis("on")
        ax_kdiff_isn.set_frame_on(False)
        ax_kdiff_isn.set_xticks([])
        ax_kdiff_isn.set_yticks([])
        ax_kdiff_isn.set_title("")
        ax_kdiff_isn.set_ylabel("ON - OFF")
        ax_kon_isn_v = fig.add_subplot(gs_4[:1, 3:6])
        ax_koff_isn_v = fig.add_subplot(gs_4[1:2, 3:6])
        ax_kdiff_isn_v = fig.add_subplot(gs_4[2:3, 3:6])
        kernels[1].plot_raw(ax_in=np.array((ax_kon_isn_v, ax_koff_isn_v, ax_kdiff_isn_v)), 
                            type="valley")
        ax_kon_isn_v.set_title("")
        ax_koff_isn_v.set_title("")
        ax_kdiff_isn_v.set_title("")

        # Orisf Kernels.
        gs_5 = fig.add_gridspec(6, 6, hspace=1, wspace=1, left=0.44, right=0.60, top=0.48, bottom=0.08)
        ax_kern_csg = fig.add_subplot(gs_5[:3, :3])
        t_peak = kernels[2].plot_raw(ax_in=ax_kern_csg, return_t=True)
        ax_kern_csg.set_xticklabels([])
        ax_kern_csg.set_xlabel("")
        ax_kern_csg_v = fig.add_subplot(gs_5[:3, 3:6])
        t_valley = kernels[2].plot_raw(type="valley", ax_in=ax_kern_csg_v, return_t=True)
        ax_kern_csg_v.set_xticklabels([])
        ax_kern_csg_v.set_xlabel("")
        ax_kern_csg_v.set_yticklabels([])
        ax_kern_csg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax_psth_csg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax_psth_csg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        ax_kern_isg = fig.add_subplot(gs_5[3:6, :3])
        t_peak = kernels[3].plot_raw(ax_in=ax_kern_isg, return_t=True)
        xticklabels = ax_kern_isg.get_xticklabels()
        ax_kern_isg.set_xticklabels(xticklabels, rotation=45)
        ax_kern_isg_v = fig.add_subplot(gs_5[3:6, 3:6])
        t_valley = kernels[3].plot_raw(type="valley", ax_in=ax_kern_isg_v, return_t=True)
        xticklabels = ax_kern_isg_v.get_xticklabels()
        ax_kern_isg_v.set_xticklabels(xticklabels, rotation=45)
        ax_kern_isg_v.set_yticklabels([])
        ax_kern_isg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax_psth_isg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax_psth_isg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        # fig.suptitle("Unit #" + str(unit.ID))
        plt.show(block=False)
        fig.set_size_inches(30, 15)
        if save_path != "":
            try:
                fig.savefig(save_path, bbox_inches="tight")
            except:
                pass



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

