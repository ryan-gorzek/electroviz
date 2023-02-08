
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

        # Waveforms.
        gs_n1 = fig.add_gridspec(4, 3, hspace=1, wspace=1, left=0.04, right=0.20, top=0.93, bottom=0.65)
        ax_waveforms = fig.add_subplot(gs_n1[:4, :3])
        unit.plot_waveforms(ax_in=ax_waveforms)
        ax_waveforms.set_title("Waveforms")
        ax_waveforms.set_yticklabels([])
        ax_waveforms.set_ylabel("")
        ax_waveforms.set_yticks([])
        
        # Optotagging Raster.
        gs_0 = fig.add_gridspec(9, 3, hspace=1, wspace=1, left=0.04, right=0.20, top=0.60, bottom=0.08)
        ax_opto_raster = fig.add_subplot(gs_0[0:5, :3])
        unit.plot_raster(stimuli[5], ax_in=ax_opto_raster)
        ax_opto_raster.set_title("Optogenetic Stimulus Responses")
        ax_opto_raster.set_xlim((-150, 7649))
        ax_opto_raster.set_xticklabels([])
        ax_opto_raster.set_xlabel("")
        ax_opto_raster.set_ylim((0, 419))
        ax_opto_raster.set_yticklabels([])
        ax_opto_raster.set_ylabel("")
        ax_opto_raster.set_yticks([])
        # Optotagging PSTH.
        ax_opto_psth = fig.add_subplot(gs_0[5:9, :3])
        unit.plot_PSTH(stimuli[5], ax_in=ax_opto_psth)
        ax_opto_psth.set_xlim((-0.75, 51.75))
        ax_opto_psth.set_yticks([])
        ax_opto_psth.set_yticklabels([])
        ax_opto_psth.set_ylabel("")

        # Unit Raster.
        gs_1 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.22, right=0.38, top=0.93, bottom=0.08)
        ax_raster = fig.add_subplot(gs_1[:13, :3])
        unit.plot_raster(list(chain(*stimuli[:4])), ax_in=ax_raster)
        ax_raster.set_title("Visual Stimulus Events")
        ax_raster.set_xlim((-150, 7649))
        ax_raster.set_ylim((0, 14319))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        stim_len = [len(stim) for stim in stimuli[:4]]
        base_y = 0.08
        for stim, color in zip(stimuli[3::-1], colors[3::-1]):
            height = 0.85 * (len(stim) / sum(stim_len))
            fig.patches.extend([Rectangle((0.3805, base_y), 0.005, height, fill=True, color=color, transform=fig.transFigure, figure=fig)])
            base_y += height
        ax_raster.set_yticklabels([])
        ax_raster.set_ylabel("")
        ax_raster.set_yticks([])

        # Stimulus PSTHs.
        gs_2 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.42, right=0.58, top=0.93, bottom=0.08)
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
        # ON_color, OFF_color = (1, 0.4, 0.4, 0.8), (0, 0.67, 0.8, 0.8)
        gs_3 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.62, right=0.78, top=0.95, bottom=0.73)
        ax_kon_csn = fig.add_subplot(gs_3[:1, :3])
        ax_koff_csn = fig.add_subplot(gs_3[1:2, :3])
        ax_kdiff_csn = fig.add_subplot(gs_3[2:3, :3])
        ON_t_peak, OFF_t_peak = kernels[0].plot_raw(ax_in=np.array((ax_kon_csn, ax_koff_csn, ax_kdiff_csn)), return_t=True)
        ax_kon_csn.axis("on")
        ax_kon_csn.set_frame_on(False)
        ax_kon_csn.set_xticks([])
        ax_kon_csn.set_yticks([])
        # ax_kon_csn.set_title("Peak")
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
        # ax_kon_csn_v = fig.add_subplot(gs_3[:1, 3:6])
        # ax_koff_csn_v = fig.add_subplot(gs_3[1:2, 3:6])
        # ax_kdiff_csn_v = fig.add_subplot(gs_3[2:3, 3:6])
        # ON_t_valley, OFF_t_valley = kernels[0].plot_raw(ax_in=np.array((ax_kon_csn_v, ax_koff_csn_v, ax_kdiff_csn_v)), 
        #                                                 type="valley", return_t=True)
        # ax_kon_csn_v.set_title("Valley")
        # ax_koff_csn_v.set_title("")
        # ax_kdiff_csn_v.set_title("")
        if ON_t_peak is not None:
            y_peak_on = np.max(ylims) - 0.075*np.max(ylims)
            y_peak_off = np.max(ylims) - 0.025*np.max(ylims)
            ax_psth_csn.text(ON_t_peak - 0.5, y_peak_on, "^")
            ax_psth_csn.text(OFF_t_peak - 0.5, y_peak_off, "^", rotation=180)
        # if ON_t_valley is not None:
        #     y_valley = np.max(ylims) - 0.025*np.max(ylims)
        #     ax_psth_csn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
        #     ax_psth_csn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

        # Ipsi Sparse Noise Kernels.
        gs_4 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.62, right=0.78, top=0.72, bottom=0.5)
        ax_kon_isn = fig.add_subplot(gs_4[:1, :3])
        ax_koff_isn = fig.add_subplot(gs_4[1:2, :3])
        ax_kdiff_isn = fig.add_subplot(gs_4[2:3, :3])
        ON_t_peak, OFF_t_peak = kernels[1].plot_raw(ax_in=np.array((ax_kon_isn, ax_koff_isn, ax_kdiff_isn)), 
                                                        return_t=True)
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
        # ax_kon_isn_v = fig.add_subplot(gs_4[:1, 3:6])
        # ax_koff_isn_v = fig.add_subplot(gs_4[1:2, 3:6])
        # ax_kdiff_isn_v = fig.add_subplot(gs_4[2:3, 3:6])
        # ON_t_valley, OFF_t_valley = kernels[1].plot_raw(ax_in=np.array((ax_kon_isn_v, ax_koff_isn_v, ax_kdiff_isn_v)), 
        #                                                 type="valley", return_t=True)
        # ax_kon_isn_v.set_title("")
        # ax_koff_isn_v.set_title("")
        # ax_kdiff_isn_v.set_title("")
        if ON_t_peak is not None:
            y_peak_on = np.max(ylims) - 0.075*np.max(ylims)
            y_peak_off = np.max(ylims) - 0.025*np.max(ylims)
            ax_psth_isn.text(ON_t_peak - 0.5, y_peak_on, "^")
            ax_psth_isn.text(OFF_t_peak - 0.5, y_peak_off, "^", rotation=180)
        # if ON_t_valley is not None:
        #     y_valley = np.max(ylims) - 0.025*np.max(ylims)
        #     ax_psth_isn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
        #     ax_psth_isn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

        # Orisf Kernels.
        gs_5 = fig.add_gridspec(6, 6, hspace=1, wspace=1, left=0.62, right=0.78, top=0.48, bottom=0.08)
        ax_kern_csg = fig.add_subplot(gs_5[:3, :3])
        t_peak = kernels[2].plot_raw(ax_in=ax_kern_csg, return_t=True)
        ax_kern_csg.set_xticklabels([])
        ax_kern_csg.set_xlabel("")
        # ax_kern_csg_v = fig.add_subplot(gs_5[:3, 3:6])
        # t_valley = kernels[2].plot_raw(type="valley", ax_in=ax_kern_csg_v, return_t=True)
        # ax_kern_csg_v.set_xticklabels([])
        # ax_kern_csg_v.set_xlabel("")
        # ax_kern_csg_v.set_yticklabels([])
        # ax_kern_csg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax_psth_csg.text(t_peak - 0.5, y_peak, "*")
        # if t_valley is not None:
        #     y_valley = np.max(ylims) - 0.025*np.max(ylims)
        #     ax_psth_csg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        ax_kern_isg = fig.add_subplot(gs_5[3:6, :3])
        t_peak = kernels[3].plot_raw(ax_in=ax_kern_isg, return_t=True)
        xticklabels = ax_kern_isg.get_xticklabels()
        ax_kern_isg.set_xticklabels(xticklabels, rotation=45)
        # ax_kern_isg_v = fig.add_subplot(gs_5[3:6, 3:6])
        # t_valley = kernels[3].plot_raw(type="valley", ax_in=ax_kern_isg_v, return_t=True)
        # xticklabels = ax_kern_isg_v.get_xticklabels()
        # ax_kern_isg_v.set_xticklabels(xticklabels, rotation=45)
        # ax_kern_isg_v.set_yticklabels([])
        # ax_kern_isg_v.set_ylabel("")
        # Add times to PSTH.
        if t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax_psth_isg.text(t_peak - 0.5, y_peak, "*")
        # if t_valley is not None:
        #     y_valley = np.max(ylims) - 0.025*np.max(ylims)
        #     ax_psth_isg.text(t_valley - 0.5, y_valley, "^", rotation=180)

        plt.show(block=False)
        fig.set_size_inches(30, 20)
        if save_path != "":
            fig.savefig(save_path + ".svg", bbox_inches="tight")
            fig.savefig(save_path + ".png", bbox_inches="tight")



class PairSummary:
    """

    """


    def __new__(
            self, 
            units, 
            stimuli, 
            kernels, 
            save_path="", 
            drop_stim=None, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig = plt.figure()

        #### Unit 0

        # Unit Raster.
        gs_1 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.01, right=0.10, top=0.93, bottom=0.08)
        ax0_raster = fig.add_subplot(gs_1[:13, :3])
        units[0].plot_raster(list(chain(*stimuli[:4])), ax_in=ax0_raster)
        ax0_raster.set_title("Visual Stimulus Events")
        ax0_raster.set_xlim((-150, 7649))
        ax0_raster.set_ylim((0, 14319))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        stim_len = [len(stim) for stim in stimuli[:4]]
        base_y = 0.08
        for stim, color in zip(stimuli[3::-1], colors[3::-1]):
            height = 0.85 * (len(stim) / sum(stim_len))
            fig.patches.extend([Rectangle((0.1005, base_y), 0.005, height, fill=True, color=color, transform=fig.transFigure, figure=fig)])
            base_y += height
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
        ON_color, OFF_color = (1, 0.4, 0.4, 0.8), (0, 0.67, 0.8, 0.8)
        gs_3 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.25, right=0.41, top=0.95, bottom=0.73)
        ax0_kon_csn = fig.add_subplot(gs_3[:1, :3])
        ax0_koff_csn = fig.add_subplot(gs_3[1:2, :3])
        ax0_kdiff_csn = fig.add_subplot(gs_3[2:3, :3])
        ON_t_peak, OFF_t_peak = kernels[0][0].plot_raw(ax_in=np.array((ax0_kon_csn, ax0_koff_csn, ax0_kdiff_csn)), return_t=True)
        ax0_kon_csn.axis("on")
        ax0_kon_csn.set_frame_on(False)
        ax0_kon_csn.set_xticks([])
        ax0_kon_csn.set_yticks([])
        ax0_kon_csn.set_title("Peak")
        ax0_kon_csn.set_ylabel("ON", color=ON_color)
        ax0_koff_csn.axis("on")
        ax0_koff_csn.set_frame_on(False)
        ax0_koff_csn.set_xticks([])
        ax0_koff_csn.set_yticks([])
        ax0_koff_csn.set_title("")
        ax0_koff_csn.set_ylabel("OFF", color=OFF_color)
        ax0_kdiff_csn.axis("on")
        ax0_kdiff_csn.set_frame_on(False)
        ax0_kdiff_csn.set_xticks([])
        ax0_kdiff_csn.set_yticks([])
        ax0_kdiff_csn.set_title("")
        ax0_kdiff_csn.set_ylabel("ON - OFF")
        ax0_kon_csn_v = fig.add_subplot(gs_3[:1, 3:6])
        ax0_koff_csn_v = fig.add_subplot(gs_3[1:2, 3:6])
        ax0_kdiff_csn_v = fig.add_subplot(gs_3[2:3, 3:6])
        ON_t_valley, OFF_t_valley = kernels[0][0].plot_raw(ax_in=np.array((ax0_kon_csn_v, ax0_koff_csn_v, ax0_kdiff_csn_v)), 
                                                          type="valley", return_t=True)
        ax0_kon_csn_v.set_title("Valley")
        ax0_koff_csn_v.set_title("")
        ax0_kdiff_csn_v.set_title("")
        if ON_t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax0_psth_csn.text(ON_t_peak - 0.5, y_peak, "^", color=ON_color)
            ax0_psth_csn.text(OFF_t_peak - 0.5, y_peak, "^", color=OFF_color)
        if ON_t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax0_psth_csn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
            ax0_psth_csn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

        # Ipsi Sparse Noise Kernels.
        gs_4 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.25, right=0.41, top=0.72, bottom=0.5)
        ax0_kon_isn = fig.add_subplot(gs_4[:1, :3])
        ax0_koff_isn = fig.add_subplot(gs_4[1:2, :3])
        ax0_kdiff_isn = fig.add_subplot(gs_4[2:3, :3])
        ON_t_peak, OFF_t_peak = kernels[0][1].plot_raw(ax_in=np.array((ax0_kon_isn, ax0_koff_isn, ax0_kdiff_isn)), return_t=True)
        ax0_kon_isn.axis("on")
        ax0_kon_isn.set_frame_on(False)
        ax0_kon_isn.set_xticks([])
        ax0_kon_isn.set_yticks([])
        ax0_kon_isn.set_title("")
        ax0_kon_isn.set_ylabel("ON", color=ON_color)
        ax0_koff_isn.axis("on")
        ax0_koff_isn.set_frame_on(False)
        ax0_koff_isn.set_xticks([])
        ax0_koff_isn.set_yticks([])
        ax0_koff_isn.set_title("")
        ax0_koff_isn.set_ylabel("OFF", color=OFF_color)
        ax0_kdiff_isn.axis("on")
        ax0_kdiff_isn.set_frame_on(False)
        ax0_kdiff_isn.set_xticks([])
        ax0_kdiff_isn.set_yticks([])
        ax0_kdiff_isn.set_title("")
        ax0_kdiff_isn.set_ylabel("ON - OFF")
        ax0_kon_isn_v = fig.add_subplot(gs_4[:1, 3:6])
        ax0_koff_isn_v = fig.add_subplot(gs_4[1:2, 3:6])
        ax0_kdiff_isn_v = fig.add_subplot(gs_4[2:3, 3:6])
        ON_t_valley, OFF_t_valley = kernels[0][1].plot_raw(ax_in=np.array((ax0_kon_isn_v, ax0_koff_isn_v, ax0_kdiff_isn_v)), 
                                                           type="valley", return_t=True)
        ax0_kon_isn_v.set_title("")
        ax0_koff_isn_v.set_title("")
        ax0_kdiff_isn_v.set_title("")
        if ON_t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax0_psth_isn.text(ON_t_peak - 0.5, y_peak, "^", color=ON_color)
            ax0_psth_isn.text(OFF_t_peak - 0.5, y_peak, "^", color=OFF_color)
        if ON_t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax0_psth_isn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
            ax0_psth_isn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

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
        xcorr_raw, xcorr_filt = cross_corr(units, drop_stim=drop_stim)
        ax_xcorr_raw = fig.add_subplot(gs_6[:3, :3])
        ax_xcorr_raw.bar(range(-100, 100), xcorr_raw, color=(0.9, 0.5, 0.5))
        ax_xcorr_raw.set_xlabel("")
        ax_xcorr_raw.set_xticklabels([])
        ax_xcorr_raw.set_ylabel("Raw")
        ylim = ax_xcorr_raw.get_ylim()
        ax_xcorr_raw.set_xlim([-100.75, 99.5])
        ax_xcorr_raw.set_xticks(np.linspace(-100, 99, 5))
        ax_xcorr_raw.fill([2, 2, 10, 10], ylim + ylim[::-1], color=(0.7, 0.7, 0.7, 0.5), zorder=-1)
        ax_xcorr_raw.set_ylim(ylim)
        ax_xcorr_raw.set_title("Cross-Correlation")
        ax_xcorr_filt = fig.add_subplot(gs_6[3:6, :3])
        ax_xcorr_filt.hlines(0, -100, 100, colors="k", linestyles="--")
        ax_xcorr_filt.plot(range(-100, 100), xcorr_filt, color=(0.9, 0.5, 0.5))
        ax_xcorr_filt.set_xlabel("Lag (ms)")
        ax_xcorr_filt.set_ylabel("Filtered")
        ylim = ax_xcorr_filt.get_ylim()
        ax_xcorr_filt.fill([2, 2, 10, 10], ylim + ylim[::-1], color=(0.7, 0.7, 0.7, 0.5))
        ax_xcorr_filt.set_ylim(ylim)
        ax_xcorr_filt.set_xlim([-100.75, 99.5])
        ax_xcorr_filt.set_xticks(np.linspace(-100, 99, 5))
        ax_xcorr_filt.set_xticklabels(np.linspace(-50, 50, 5))

        #### Unit 1

        # Unit Raster.
        gs_12 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.9, right=0.99, top=0.93, bottom=0.08)
        ax1_raster = fig.add_subplot(gs_12[:13, :3])
        units[1].plot_raster(list(chain(*stimuli[:4])), ax_in=ax1_raster)
        ax1_raster.set_title("Visual Stimulus Events")
        ax1_raster.set_xlim((-150, 7649))
        ax1_raster.set_ylim((0, 14319))
        colors = ((0.2, 0.2, 0.9), (0.9, 0.2, 0.2), (0.7, 0.2, 0.7), (0.9, 0.5, 0.2))
        stim_len = [len(stim) for stim in stimuli[:4]]
        base_y = 0.08
        for stim, color in zip(stimuli[3::-1], colors[3::-1]):
            height = 0.85 * (len(stim) / sum(stim_len))
            fig.patches.extend([Rectangle((0.8990, base_y), 0.005, height, fill=True, color=color, transform=fig.transFigure, figure=fig)])
            base_y += height
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
        ON_t_peak, OFF_t_peak = kernels[1][0].plot_raw(ax_in=np.array((ax1_kon_csn, ax1_koff_csn, ax1_kdiff_csn)), return_t=True)
        ax1_kon_csn.axis("on")
        ax1_kon_csn.set_frame_on(False)
        ax1_kon_csn.set_xticks([])
        ax1_kon_csn.set_yticks([])
        ax1_kon_csn.set_title("Peak")
        ax1_kon_csn.set_ylabel("ON", color=ON_color)
        ax1_koff_csn.axis("on")
        ax1_koff_csn.set_frame_on(False)
        ax1_koff_csn.set_xticks([])
        ax1_koff_csn.set_yticks([])
        ax1_koff_csn.set_title("")
        ax1_koff_csn.set_ylabel("OFF", color=OFF_color)
        ax1_kdiff_csn.axis("on")
        ax1_kdiff_csn.set_frame_on(False)
        ax1_kdiff_csn.set_xticks([])
        ax1_kdiff_csn.set_yticks([])
        ax1_kdiff_csn.set_title("")
        ax1_kdiff_csn.set_ylabel("ON - OFF")
        ax1_kon_csn_v = fig.add_subplot(gs_10[:1, 3:6])
        ax1_koff_csn_v = fig.add_subplot(gs_10[1:2, 3:6])
        ax1_kdiff_csn_v = fig.add_subplot(gs_10[2:3, 3:6])
        ON_t_valley, OFF_t_valley = kernels[1][0].plot_raw(ax_in=np.array((ax1_kon_csn_v, ax1_koff_csn_v, ax1_kdiff_csn_v)), 
                                                           type="valley", return_t=True)
        ax1_kon_csn_v.set_title("Valley")
        ax1_koff_csn_v.set_title("")
        ax1_kdiff_csn_v.set_title("")
        if ON_t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax1_psth_csn.text(ON_t_peak - 0.5, y_peak, "^", color=ON_color)
            ax1_psth_csn.text(OFF_t_peak - 0.5, y_peak, "^", color=OFF_color)
        if ON_t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax1_psth_csn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
            ax1_psth_csn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

        # Ipsi Sparse Noise Kernels.
        gs_9 = fig.add_gridspec(3, 6, hspace=0.01, wspace=1, left=0.59, right=0.75, top=0.72, bottom=0.5)
        ax1_kon_isn = fig.add_subplot(gs_9[:1, :3])
        ax1_koff_isn = fig.add_subplot(gs_9[1:2, :3])
        ax1_kdiff_isn = fig.add_subplot(gs_9[2:3, :3])
        ON_t_peak, OFF_t_peak = kernels[1][1].plot_raw(ax_in=np.array((ax1_kon_isn, ax1_koff_isn, ax1_kdiff_isn)), return_t=True)
        ax1_kon_isn.axis("on")
        ax1_kon_isn.set_frame_on(False)
        ax1_kon_isn.set_xticks([])
        ax1_kon_isn.set_yticks([])
        ax1_kon_isn.set_title("")
        ax1_kon_isn.set_ylabel("ON", color=ON_color)
        ax1_koff_isn.axis("on")
        ax1_koff_isn.set_frame_on(False)
        ax1_koff_isn.set_xticks([])
        ax1_koff_isn.set_yticks([])
        ax1_koff_isn.set_title("")
        ax1_koff_isn.set_ylabel("OFF", color=OFF_color)
        ax1_kdiff_isn.axis("on")
        ax1_kdiff_isn.set_frame_on(False)
        ax1_kdiff_isn.set_xticks([])
        ax1_kdiff_isn.set_yticks([])
        ax1_kdiff_isn.set_title("")
        ax1_kdiff_isn.set_ylabel("ON - OFF")
        ax1_kon_isn_v = fig.add_subplot(gs_9[:1, 3:6])
        ax1_koff_isn_v = fig.add_subplot(gs_9[1:2, 3:6])
        ax1_kdiff_isn_v = fig.add_subplot(gs_9[2:3, 3:6])
        ON_t_valley, OFF_t_valley = kernels[1][1].plot_raw(ax_in=np.array((ax1_kon_isn_v, ax1_koff_isn_v, ax1_kdiff_isn_v)), 
                                                           type="valley", return_t=True)
        ax1_kon_isn_v.set_title("")
        ax1_koff_isn_v.set_title("")
        ax1_kdiff_isn_v.set_title("")
        if ON_t_peak is not None:
            y_peak = np.max(ylims) - 0.075*np.max(ylims)
            ax1_psth_isn.text(ON_t_peak - 0.5, y_peak, "^", color=ON_color)
            ax1_psth_isn.text(OFF_t_peak - 0.5, y_peak, "^", color=OFF_color)
        if ON_t_valley is not None:
            y_valley = np.max(ylims) - 0.025*np.max(ylims)
            ax1_psth_isn.text(ON_t_valley - 0.5, y_valley, "^", color=ON_color, rotation=180)
            ax1_psth_isn.text(OFF_t_valley - 0.5, y_valley, "^", color=OFF_color, rotation=180)

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

