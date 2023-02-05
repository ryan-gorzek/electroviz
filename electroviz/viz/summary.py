
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
        ax_raster.set_xlim((-150, 7649))
        ax_raster.set_ylim((0, 14839))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        stim_len = [len(stim) for stim in stimuli[:4]]
        base_y = 0.08
        for stim, color in zip(stimuli[3::-1], colors[3::-1]):
            height = 0.85 * (len(stim) / sum(stim_len))
            fig.patches.extend([Rectangle((0.201, base_y), 0.005, height, fill=True, color=color, transform=fig.transFigure, figure=fig)])
            base_y += height
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

        plt.show(block=False)
        fig.set_size_inches(30, 15)
        if save_path != "":
            fig.savefig(save_path, bbox_inches="tight")



class PairSummary:
    """

    """


    def __new__(
            self, 
            units, 
            stimuli, 
            kernels, 
            save_path="", 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig = plt.figure()

        #### Unit 0

        # Unit Raster.
        gs_1 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.01, right=0.10, top=0.93, bottom=0.08)
        ax0_raster = fig.add_subplot(gs_1[:13, :3])
        units[0].plot_raster(list(chain(*stimuli)), ax_in=ax0_raster)
        ax0_raster.set_title("Stimulus Events")
        ax0_raster.set_xlim((-1.25, 53.25))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        prev_len = 0
        for stim, color in zip(stimuli, colors):
            stim_len = len(stim)
            ax0_raster.add_patch(Rectangle((51.25, prev_len), 2, stim_len, color=color))
            prev_len += stim_len
        ax0_raster.spines["right"].set_visible(False)
        ax0_raster.set_yticklabels([])
        ax0_raster.set_ylabel("")
        ax0_raster.set_yticks([])

        # Stimulus PSTHs.
        gs_2 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.14, right=0.22, top=0.93, bottom=0.08)
        ax0_psth_csn = fig.add_subplot(gs_2[:3, :3])
        units[0].plot_PSTH(stimuli[0], ax_in=ax0_psth_csn)
        ax0_psth_csn.set_xticklabels([])
        ax0_psth_csn.set_xlabel("")
        ax0_psth_csn.set_xlim((-0.75, 51.75))
        ax0_psth_csn.set_ylabel("Contra Sparse Noise", color=colors[0])
        ax0_psth_csn.set_title("Spike Rate (Hz)")

        ax0_psth_isn = fig.add_subplot(gs_2[3:6, :3])
        units[0].plot_PSTH(stimuli[1], ax_in=ax0_psth_isn)
        ax0_psth_isn.set_xticklabels([])
        ax0_psth_isn.set_xlabel("")
        ax0_psth_isn.set_xlim((-0.75, 51.75))
        ax0_psth_isn.set_ylabel("Ipsi Sparse Noise", color=colors[1])

        ax0_psth_csg = fig.add_subplot(gs_2[6:9, :3])
        units[0].plot_PSTH(stimuli[2], ax_in=ax0_psth_csg)
        ax0_psth_csg.set_xticklabels([])
        ax0_psth_csg.set_xlabel("")
        ax0_psth_csg.set_xlim((-0.75, 51.75))
        ax0_psth_csg.set_ylabel("Contra Static Gratings", color=colors[2])

        ax0_psth_isg = fig.add_subplot(gs_2[9:13, :3])
        units[0].plot_PSTH(stimuli[3], ax_in=ax0_psth_isg)
        ax0_psth_isg.set_xlim((-0.75, 51.75))
        ax0_psth_isg.set_ylabel("Ipsi Static Gratings", color=colors[3])

        ylims = []
        for ax in [ax0_psth_csn, ax0_psth_isn, ax0_psth_csg, ax0_psth_isg]:
            ylim = ax.get_ylim()
            ylims.append(ylim[1])
        for ax in [ax0_psth_csn, ax0_psth_isn, ax0_psth_csg, ax0_psth_isg]:
            ax.set_ylim((0, np.max(ylims)))

        # Contra Sparse Noise Kernels.
        gs_3 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.25, right=0.41, top=0.95, bottom=0.73)
        ax0_kon_csn = fig.add_subplot(gs_3[:1, :3])
        ax0_koff_csn = fig.add_subplot(gs_3[1:2, :3])
        ax0_kdiff_csn = fig.add_subplot(gs_3[2:3, :3])
        kernels[0][0].plot_raw(ax_in=np.array((ax0_kon_csn, ax0_koff_csn, ax0_kdiff_csn)))
        ax0_kon_csn.axis("on")
        ax0_kon_csn.set_frame_on(False)
        ax0_kon_csn.set_xticks([])
        ax0_kon_csn.set_yticks([])
        ax0_kon_csn.set_title("Peak")
        ax0_kon_csn.set_ylabel("ON")
        ax0_koff_csn.axis("on")
        ax0_koff_csn.set_frame_on(False)
        ax0_koff_csn.set_xticks([])
        ax0_koff_csn.set_yticks([])
        ax0_koff_csn.set_title("")
        ax0_koff_csn.set_ylabel("OFF")
        ax0_kdiff_csn.axis("on")
        ax0_kdiff_csn.set_frame_on(False)
        ax0_kdiff_csn.set_xticks([])
        ax0_kdiff_csn.set_yticks([])
        ax0_kdiff_csn.set_title("")
        ax0_kdiff_csn.set_ylabel("ON - OFF")
        ax0_kon_csn_v = fig.add_subplot(gs_3[:1, 3:6])
        ax0_koff_csn_v = fig.add_subplot(gs_3[1:2, 3:6])
        ax0_kdiff_csn_v = fig.add_subplot(gs_3[2:3, 3:6])
        kernels[0][0].plot_raw(ax_in=np.array((ax0_kon_csn_v, ax0_koff_csn_v, ax0_kdiff_csn_v)), 
                               type="valley")
        ax0_kon_csn_v.set_title("Valley")
        ax0_koff_csn_v.set_title("")
        ax0_kdiff_csn_v.set_title("")

        # Ipsi Sparse Noise Kernels.
        gs_4 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.25, right=0.41, top=0.72, bottom=0.5)
        ax0_kon_isn = fig.add_subplot(gs_4[:1, :3])
        ax0_koff_isn = fig.add_subplot(gs_4[1:2, :3])
        ax0_kdiff_isn = fig.add_subplot(gs_4[2:3, :3])
        kernels[0][1].plot_raw(ax_in=np.array((ax0_kon_isn, ax0_koff_isn, ax0_kdiff_isn)))
        ax0_kon_isn.axis("on")
        ax0_kon_isn.set_frame_on(False)
        ax0_kon_isn.set_xticks([])
        ax0_kon_isn.set_yticks([])
        ax0_kon_isn.set_title("")
        ax0_kon_isn.set_ylabel("ON")
        ax0_koff_isn.axis("on")
        ax0_koff_isn.set_frame_on(False)
        ax0_koff_isn.set_xticks([])
        ax0_koff_isn.set_yticks([])
        ax0_koff_isn.set_title("")
        ax0_koff_isn.set_ylabel("OFF")
        ax0_kdiff_isn.axis("on")
        ax0_kdiff_isn.set_frame_on(False)
        ax0_kdiff_isn.set_xticks([])
        ax0_kdiff_isn.set_yticks([])
        ax0_kdiff_isn.set_title("")
        ax0_kdiff_isn.set_ylabel("ON - OFF")
        ax0_kon_isn_v = fig.add_subplot(gs_4[:1, 3:6])
        ax0_koff_isn_v = fig.add_subplot(gs_4[1:2, 3:6])
        ax0_kdiff_isn_v = fig.add_subplot(gs_4[2:3, 3:6])
        kernels[0][1].plot_raw(ax_in=np.array((ax0_kon_isn_v, ax0_koff_isn_v, ax0_kdiff_isn_v)), 
                            type="valley")
        ax0_kon_isn_v.set_title("")
        ax0_koff_isn_v.set_title("")
        ax0_kdiff_isn_v.set_title("")

        # Orisf Kernels.
        gs_5 = fig.add_gridspec(6, 6, hspace=1, wspace=1, left=0.25, right=0.41, top=0.48, bottom=0.08)
        ax0_kern_csg = fig.add_subplot(gs_5[:3, :3])
        t_peak = kernels[0][2].plot_raw(ax_in=ax0_kern_csg, return_t=True)
        ax0_kern_csg.set_xticklabels([])
        ax0_kern_csg.set_xlabel("")
        ax0_kern_csg_v = fig.add_subplot(gs_5[:3, 3:6])
        t_valley = kernels[0][2].plot_raw(type="valley", ax_in=ax0_kern_csg_v, return_t=True)
        ax0_kern_csg_v.set_xticklabels([])
        ax0_kern_csg_v.set_xlabel("")
        ax0_kern_csg_v.set_yticklabels([])
        ax0_kern_csg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax0_psth_csg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax0_psth_csg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        ax0_kern_isg = fig.add_subplot(gs_5[3:6, :3])
        t_peak = kernels[0][3].plot_raw(ax_in=ax0_kern_isg, return_t=True)
        xticklabels = ax0_kern_isg.get_xticklabels()
        ax0_kern_isg.set_xticklabels(xticklabels, rotation=45)
        ax0_kern_isg_v = fig.add_subplot(gs_5[3:6, 3:6])
        t_valley = kernels[0][3].plot_raw(type="valley", ax_in=ax0_kern_isg_v, return_t=True)
        xticklabels = ax0_kern_isg_v.get_xticklabels()
        ax0_kern_isg_v.set_xticklabels(xticklabels, rotation=45)
        ax0_kern_isg_v.set_yticklabels([])
        ax0_kern_isg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax0_psth_isg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax0_psth_isg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        #### Cross-Correlation

        gs_6 = fig.add_gridspec(6, 3, hspace=1, wspace=1, left=0.45, right=0.55, top=0.75, bottom=0.28)
        xcorr_raw, xcorr_filt = cross_corr(units)
        ax_xcorr_raw = fig.add_subplot(gs_6[:3, :3])
        ax_xcorr_raw.bar(range(-50, 51), xcorr_raw, color=(0.9, 0.5, 0.5))
        ax_xcorr_raw.set_xlabel("")
        ax_xcorr_raw.set_xticklabels([])
        ax_xcorr_raw.set_ylabel("Raw")
        ylim = ax_xcorr_raw.get_ylim()
        ax_xcorr_raw.set_xlim([-50.75, 50.75])
        ax_xcorr_raw.set_xticks(np.linspace(-50, 50, 5))
        ax_xcorr_raw.fill([1, 1, 5, 5], ylim + ylim[::-1], color=(0.7, 0.7, 0.7, 0.5), zorder=-1)
        ax_xcorr_raw.set_ylim(ylim)
        ax_xcorr_raw.set_title("Cross-Correlation")
        ax_xcorr_filt = fig.add_subplot(gs_6[3:6, :3])
        ax_xcorr_filt.hlines(0, -50, 51, colors="k", linestyles="--")
        ax_xcorr_filt.plot(range(-50, 51), xcorr_filt, color=(0.9, 0.5, 0.5))
        ax_xcorr_filt.set_xlabel("Lag (ms)")
        ax_xcorr_filt.set_ylabel("Filtered")
        ylim = ax_xcorr_filt.get_ylim()
        ax_xcorr_filt.fill([1, 1, 5, 5], ylim + ylim[::-1], color=(0.7, 0.7, 0.7, 0.5))
        ax_xcorr_filt.set_ylim(ylim)
        ax_xcorr_filt.set_xlim([-50.75, 50.75])
        ax_xcorr_filt.set_xticks(np.linspace(-50, 50, 5))

        #### Unit 1

        # Unit Raster.
        gs_12 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.9, right=0.99, top=0.93, bottom=0.08)
        ax1_raster = fig.add_subplot(gs_12[:13, :3])
        units[1].plot_raster(list(chain(*stimuli)), ax_in=ax1_raster)
        ax1_raster.set_title("Stimulus Events")
        ax1_raster.set_xlim((-3.25, 51.25))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        prev_len = 0
        for stim, color in zip(stimuli, colors):
            stim_len = len(stim)
            ax1_raster.add_patch(Rectangle((-3.25, prev_len), 2, stim_len, color=color))
            prev_len += stim_len
        ax1_raster.spines["left"].set_visible(False)
        ax1_raster.set_yticklabels([])
        ax1_raster.set_ylabel("")
        ax1_raster.set_yticks([])

        # Stimulus PSTHs.
        gs_11 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.78, right=0.86, top=0.93, bottom=0.08)
        ax1_psth_csn = fig.add_subplot(gs_11[:3, :3])
        units[1].plot_PSTH(stimuli[0], ax_in=ax1_psth_csn)
        ax1_psth_csn.set_xticklabels([])
        ax1_psth_csn.set_xlabel("")
        ax1_psth_csn.set_xlim((-0.75, 51.75))
        ax1_psth_csn.set_ylabel("Contra Sparse Noise", color=colors[0])
        ax1_psth_csn.set_title("Spike Rate (Hz)")
        ax1_psth_csn.yaxis.tick_right()

        ax1_psth_isn = fig.add_subplot(gs_11[3:6, :3])
        units[1].plot_PSTH(stimuli[1], ax_in=ax1_psth_isn)
        ax1_psth_isn.set_xticklabels([])
        ax1_psth_isn.set_xlabel("")
        ax1_psth_isn.set_xlim((-0.75, 51.75))
        ax1_psth_isn.set_ylabel("Ipsi Sparse Noise", color=colors[1])
        ax1_psth_isn.yaxis.tick_right()

        ax1_psth_csg = fig.add_subplot(gs_11[6:9, :3])
        units[1].plot_PSTH(stimuli[2], ax_in=ax1_psth_csg)
        ax1_psth_csg.set_xticklabels([])
        ax1_psth_csg.set_xlabel("")
        ax1_psth_csg.set_xlim((-0.75, 51.75))
        ax1_psth_csg.set_ylabel("Contra Static Gratings", color=colors[2])
        ax1_psth_csg.yaxis.tick_right()

        ax1_psth_isg = fig.add_subplot(gs_11[9:13, :3])
        units[1].plot_PSTH(stimuli[3], ax_in=ax1_psth_isg)
        ax1_psth_isg.set_xlim((-0.75, 51.75))
        ax1_psth_isg.set_ylabel("Ipsi Static Gratings", color=colors[3])
        ax1_psth_isg.yaxis.tick_right()

        ylims = []
        for ax in [ax1_psth_csn, ax1_psth_isn, ax1_psth_csg, ax1_psth_isg]:
            ylim = ax.get_ylim()
            ylims.append(ylim[1])
        for ax in [ax1_psth_csn, ax1_psth_isn, ax1_psth_csg, ax1_psth_isg]:
            ax.set_ylim((0, np.max(ylims)))

        # Contra Sparse Noise Kernels.
        gs_10 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.59, right=0.75, top=0.95, bottom=0.73)
        ax1_kon_csn = fig.add_subplot(gs_10[:1, :3])
        ax1_koff_csn = fig.add_subplot(gs_10[1:2, :3])
        ax1_kdiff_csn = fig.add_subplot(gs_10[2:3, :3])
        kernels[1][0].plot_raw(ax_in=np.array((ax1_kon_csn, ax1_koff_csn, ax1_kdiff_csn)))
        ax1_kon_csn.axis("on")
        ax1_kon_csn.set_frame_on(False)
        ax1_kon_csn.set_xticks([])
        ax1_kon_csn.set_yticks([])
        ax1_kon_csn.set_title("Peak")
        ax1_kon_csn.set_ylabel("ON")
        ax1_koff_csn.axis("on")
        ax1_koff_csn.set_frame_on(False)
        ax1_koff_csn.set_xticks([])
        ax1_koff_csn.set_yticks([])
        ax1_koff_csn.set_title("")
        ax1_koff_csn.set_ylabel("OFF")
        ax1_kdiff_csn.axis("on")
        ax1_kdiff_csn.set_frame_on(False)
        ax1_kdiff_csn.set_xticks([])
        ax1_kdiff_csn.set_yticks([])
        ax1_kdiff_csn.set_title("")
        ax1_kdiff_csn.set_ylabel("ON - OFF")
        ax1_kon_csn_v = fig.add_subplot(gs_10[:1, 3:6])
        ax1_koff_csn_v = fig.add_subplot(gs_10[1:2, 3:6])
        ax1_kdiff_csn_v = fig.add_subplot(gs_10[2:3, 3:6])
        kernels[1][0].plot_raw(ax_in=np.array((ax1_kon_csn_v, ax1_koff_csn_v, ax1_kdiff_csn_v)), 
                            type="valley")
        ax1_kon_csn_v.set_title("Valley")
        ax1_koff_csn_v.set_title("")
        ax1_kdiff_csn_v.set_title("")

        # Ipsi Sparse Noise Kernels.
        gs_9 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.59, right=0.75, top=0.72, bottom=0.5)
        ax1_kon_isn = fig.add_subplot(gs_9[:1, :3])
        ax1_koff_isn = fig.add_subplot(gs_9[1:2, :3])
        ax1_kdiff_isn = fig.add_subplot(gs_9[2:3, :3])
        kernels[1][1].plot_raw(ax_in=np.array((ax1_kon_isn, ax1_koff_isn, ax1_kdiff_isn)))
        ax1_kon_isn.axis("on")
        ax1_kon_isn.set_frame_on(False)
        ax1_kon_isn.set_xticks([])
        ax1_kon_isn.set_yticks([])
        ax1_kon_isn.set_title("")
        ax1_kon_isn.set_ylabel("ON")
        ax1_koff_isn.axis("on")
        ax1_koff_isn.set_frame_on(False)
        ax1_koff_isn.set_xticks([])
        ax1_koff_isn.set_yticks([])
        ax1_koff_isn.set_title("")
        ax1_koff_isn.set_ylabel("OFF")
        ax1_kdiff_isn.axis("on")
        ax1_kdiff_isn.set_frame_on(False)
        ax1_kdiff_isn.set_xticks([])
        ax1_kdiff_isn.set_yticks([])
        ax1_kdiff_isn.set_title("")
        ax1_kdiff_isn.set_ylabel("ON - OFF")
        ax1_kon_isn_v = fig.add_subplot(gs_9[:1, 3:6])
        ax1_koff_isn_v = fig.add_subplot(gs_9[1:2, 3:6])
        ax1_kdiff_isn_v = fig.add_subplot(gs_9[2:3, 3:6])
        kernels[1][1].plot_raw(ax_in=np.array((ax1_kon_isn_v, ax1_koff_isn_v, ax1_kdiff_isn_v)), 
                            type="valley")
        ax1_kon_isn_v.set_title("")
        ax1_koff_isn_v.set_title("")
        ax1_kdiff_isn_v.set_title("")

        # Orisf Kernels.
        gs_8 = fig.add_gridspec(6, 6, hspace=1, wspace=1, left=0.59, right=0.75, top=0.48, bottom=0.08)
        ax1_kern_csg = fig.add_subplot(gs_8[:3, :3])
        t_peak = kernels[1][2].plot_raw(ax_in=ax1_kern_csg, return_t=True)
        ax1_kern_csg.set_xticklabels([])
        ax1_kern_csg.set_xlabel("")
        ax1_kern_csg_v = fig.add_subplot(gs_8[:3, 3:6])
        t_valley = kernels[1][2].plot_raw(type="valley", ax_in=ax1_kern_csg_v, return_t=True)
        ax1_kern_csg_v.set_xticklabels([])
        ax1_kern_csg_v.set_xlabel("")
        ax1_kern_csg_v.set_yticklabels([])
        ax1_kern_csg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax1_psth_csg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax1_psth_csg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        ax1_kern_isg = fig.add_subplot(gs_8[3:6, :3])
        t_peak = kernels[1][3].plot_raw(ax_in=ax1_kern_isg, return_t=True)
        xticklabels = ax1_kern_isg.get_xticklabels()
        ax1_kern_isg.set_xticklabels(xticklabels, rotation=45)
        ax1_kern_isg_v = fig.add_subplot(gs_8[3:6, 3:6])
        t_valley = kernels[1][3].plot_raw(type="valley", ax_in=ax1_kern_isg_v, return_t=True)
        xticklabels = ax1_kern_isg_v.get_xticklabels()
        ax1_kern_isg_v.set_xticklabels(xticklabels, rotation=45)
        ax1_kern_isg_v.set_yticklabels([])
        ax1_kern_isg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax1_psth_isg.text(t_peak - 0.5, y_peak, "^")
        if t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax1_psth_isg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        #### Figure Parameters

        plt.show(block=False)
        fig.set_size_inches(36, 16)
        if save_path != "":
            fig.savefig(save_path, bbox_inches="tight")

