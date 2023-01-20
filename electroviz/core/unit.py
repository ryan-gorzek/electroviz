
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from electroviz.viz.psth import PSTH
from electroviz.viz.raster import Raster

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
        gs = fig.add_gridspec(12, 15, hspace=2, wspace=2)

        ax_raster = fig.add_subplot(gs[:9, :3])
        self._Population.plot_raster(None, responses=self._Population._responses, ax_in=ax_raster)

        ax_xcorr = fig.add_subplot(gs[:9, 3:6])
        ax_xcorr.axis("off")
        ax_hist = fig.add_subplot(gs[9:13, :3])
        ax_hist.axis("off")
        ax_peaks = fig.add_subplot(gs[9:13, 3:6])
        ax_peaks.axis("off")

        ax_psth_csn = fig.add_subplot(gs[:3, 6:9])
        self.plot_PSTH(stimuli[0], ax_in=ax_psth_csn)

        ax_kon_csn = fig.add_subplot(gs[:1, 9:12])
        ax_koff_csn = fig.add_subplot(gs[1:2, 9:12])
        ax_kdiff_csn = fig.add_subplot(gs[2:3, 9:12])
        kernels[0].plot_raw(ax_in=np.array((ax_kon_csn, ax_koff_csn, ax_kdiff_csn)))

        ax_norm_csn = fig.add_subplot(gs[:3, 12:16])
        kernels[0].plot_norm_delay(ax_in=ax_norm_csn)

        ax_psth_isn = fig.add_subplot(gs[3:6, 6:9])
        self.plot_PSTH(stimuli[1], ax_in=ax_psth_isn)

        ax_kon_isn = fig.add_subplot(gs[3:4, 9:12])
        ax_koff_isn = fig.add_subplot(gs[4:5, 9:12])
        ax_kdiff_isn = fig.add_subplot(gs[5:6, 9:12])
        kernels[1].plot_raw(ax_in=np.array((ax_kon_isn, ax_koff_isn, ax_kdiff_isn)))

        ax_norm_isn = fig.add_subplot(gs[3:6, 12:16])
        kernels[1].plot_norm_delay(ax_in=ax_norm_isn)

        ax_psth_csg = fig.add_subplot(gs[6:9, 6:9])
        self.plot_PSTH(stimuli[2], ax_in=ax_psth_csg)

        ax_kern_csg = fig.add_subplot(gs[6:9, 9:12])
        kernels[2].plot_raw(ax_in=ax_kern_csg)

        ax_norm_csg = fig.add_subplot(gs[6:9, 12:16])
        kernels[2].plot_norm_delay(ax_in=ax_norm_csg)

        ax_psth_isg = fig.add_subplot(gs[9:13, 6:9])
        self.plot_PSTH(stimuli[3], ax_in=ax_psth_isg)

        ax_kern_isg = fig.add_subplot(gs[9:13, 9:12])
        kernels[3].plot_raw(ax_in=ax_kern_isg)

        ax_norm_isg = fig.add_subplot(gs[9:13, 12:16])
        kernels[3].plot_norm_delay(ax_in=ax_norm_isg)

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
        return self.spike_times[0, sample_window[0]:sample_window[1]].toarray().squeeze()

