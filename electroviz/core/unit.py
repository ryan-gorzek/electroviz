
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from electroviz.viz.psth import PSTH
from electroviz.viz.raster import Raster
from scipy.signal import correlate, correlation_lags
from itertools import chain

class Unit:
    """
    
    """


    def __init__(
            self, 
            unit_id, 
            imec_sync, 
            kilosort_spikes, 
            population, 
        ):
        """"""
        
        self.ID = unit_id
        self._Sync = imec_sync
        self._Spikes = kilosort_spikes
        self._Population = population
        self.total_samples = self._Population.total_samples
        self.sampling_rate = self._Sync.sampling_rate
        self.spike_times = np.empty((0, 0))


    def plot_PSTH(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            ax_in=None, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        PSTH(time_window, responses.mean(axis=0).squeeze(), ax_in=ax_in)


    def plot_raster(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=5, 
            zscore=False, 
        ):
        """"""

        responses = self.get_response(stimulus, time_window, bin_size=bin_size)
        Raster(time_window, responses, ylabel="Stimulus Event", z_score=zscore)


    def plot_summary(
            self, 
            stimuli, 
            kernels, 
        ):
        """"""

        mpl_use("Qt5Agg")
        fig = plt.figure()

        gs_1 = fig.add_gridspec(12, 6, hspace=1, wspace=1, left=0.04, right=0.36, top=0.93, bottom=0.08)
        ax_raster = fig.add_subplot(gs_1[:9, :3])
        self._Population.plot_raster(None, responses=self._Population._responses, ax_in=ax_raster)
        ax_raster.set_xticklabels([])
        ax_raster.set_xlabel("")
        ax_raster.set_title("Population Responses")
        ax_xcorr = fig.add_subplot(gs_1[:9, 3:6])
        # ax_xcorr.axis("off")
        ax_psth = fig.add_subplot(gs_1[9:13, :3])
        self.plot_PSTH(list(chain(*stimuli)), ax_in=ax_psth)
        ax_peaks = fig.add_subplot(gs_1[9:13, 3:6])
        # ax_peaks.axis("off")

        # Stimulus PSTHs.
        gs_2 = fig.add_gridspec(12, 3, hspace=1, wspace=1, left=0.40, right=0.56, top=0.93, bottom=0.08)
        ax_psth_csn = fig.add_subplot(gs_2[:3, :3])
        self.plot_PSTH(stimuli[0], ax_in=ax_psth_csn)
        ax_psth_csn.set_xticklabels([])
        ax_psth_csn.set_xlabel("")
        ax_psth_csn.set_ylabel("")
        ax_psth_csn.set_title("Spike Rate (Hz)")

        ax_psth_isn = fig.add_subplot(gs_2[3:6, :3])
        self.plot_PSTH(stimuli[1], ax_in=ax_psth_isn)
        ax_psth_isn.set_xticklabels([])
        ax_psth_isn.set_xlabel("")
        ax_psth_isn.set_ylabel("")

        ax_psth_csg = fig.add_subplot(gs_2[6:9, :3])
        self.plot_PSTH(stimuli[2], ax_in=ax_psth_csg)
        ax_psth_csg.set_xticklabels([])
        ax_psth_csg.set_xlabel("")
        ax_psth_csg.set_ylabel("")

        ax_psth_isg = fig.add_subplot(gs_2[9:13, :3])
        self.plot_PSTH(stimuli[3], ax_in=ax_psth_isg)
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


    def get_response(
            self, 
            stimulus, 
            time_window=(-50, 200), 
            bin_size=1, 
        ):
        """"""

        sample_window = np.array(time_window) * 30
        num_samples = int(sample_window[1] - sample_window[0])
        num_bins = int(num_samples/(bin_size * 30))
        responses = np.zeros((len(stimulus), num_bins))
        for event in stimulus:
            window = (sample_window + event.sample_onset).astype(int)
            resp = self.get_spike_times(sample_window=window)
            bin_resp = resp.reshape((num_bins, -1)).sum(axis=1) / (bin_size / 1000)
            responses[event.index, :] = bin_resp
        return responses


    def add_metric(
            self, 
            metric_name, 
            metric, 
        ):
        """"""
        
        if not metric_name in self._Population.units.columns:
            self._Population.units[metric_name] = [np.nan]*self._Population.units.shape[0]
        (unit_idx,) = np.where(self._Population.units["unit_id"] == self.ID)[0]
        self._Population.units.at[unit_idx, metric_name] = metric


    def get_spike_times(
            self, 
            sample_window=(None, None), 
        ):
        """"""
        
        if self.spike_times.shape[0] == 0:
            spike_times_matrix = self._Spikes.spike_times.tocsr()
            self.spike_times = spike_times_matrix[self.ID].tocsc()
        if (sample_window[0] is None) & (sample_window[1] is None):
            return self.spike_times[0, :].toarray().squeeze()
        else:
            return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()


    def _bin_spikes(
            self, 
            bin_size=100, 
        ):
        """"""
        
        drop_end = int(self.total_samples % (bin_size * 30))
        num_bins = int((self.total_samples - drop_end)/(bin_size * 30))
        spike_times = self.get_spike_times()
        spike_times = spike_times[:-drop_end]
        spike_rate = spike_times.reshape((num_bins, -1)).sum(axis=1) / (bin_size / 1000)
        return spike_rate

