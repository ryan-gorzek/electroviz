# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

class Unit:
    '''
    docstring
    '''
    
    def __init__(self, unit_df, unit_probe_id, unit_local_index, unit_location):
        """"""
        print('Unit')
        self.id = unit_df.index[0]
        self.info_df = unit_df.loc[[self.id], 
            ["peak_channel_id", "cluster_id"]]
        self.info_df.insert(0, "probe_id", unit_probe_id)
        self.info_df.insert(2, "local_index", unit_local_index)
        self.info_df.rename(columns={"local_index":"probe_channel_number"}, inplace=True)
        self.info_df["location"] = unit_location
        self.quality_df = unit_df.loc[[self.id], 
            ["quality", "firing_rate", "presence_ratio", "max_drift",
             "cumulative_drift", "silhouette_score", "isi_violations", 
             "amplitude_cutoff", "isolation_distance", "l_ratio", "d_prime", 
             "nn_hit_rate", "nn_miss_rate", ]]
        self.stats_df = unit_df.loc[[self.id], 
            ["firing_rate", "waveform_duration", "spread", "waveform_halfwidth",
             "snr", "PT_ratio", "repolarization_slope", "recovery_slope", 
             "velocity_above", "velocity_below"]]
        self.stats_df.rename(columns={"spread":"waveform_spread"}, inplace=True)
        self.spike_times = unit_df.at[self.id, "spike_times"]
        self.spike_amplitudes = unit_df.at[self.id, "spike_amplitudes"]
        self.mean_waveforms = unit_df.at[self.id, "waveform_mean"].T
        
    def plot_mean_waveform(self, channel="peak", color="k"):
        """"""
        if channel == "peak":
            plot_channel = self.info_df.at[self.id, "probe_channel_number"]
        time_series = np.linspace(0, 2.7, num=82)
        plt.plot(time_series, self.mean_waveforms[:, plot_channel], color=color)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential ($\mu$V)")
        # add unit id, channel (w/ peak indication), and location to lower right
        fig = plt.gcf()
        fig.set_size_inches(4, 4)
        plt.tight_layout()
        ax = plt.gca()
        ax.set_aspect(1./ax.get_data_ratio())
        return ax

    # def plot_channel_waveforms(self):
        
    def plot_spike_raster(self, bin_size_s=0.0005, binary=False):
        """"""
        times, counts, amplitudes = self._bin_spikes(bin_size_s=bin_size_s)
        plt.imshow(np.expand_dims(amplitudes, axis=0), 
                   aspect=0.2*amplitudes.size, 
                   cmap="gray",
                   clim=[0, np.nanmean(amplitudes)])
        plt.xlabel("Time (s)")
        plt.ylabel("Spike Amplitude (?)")
        # add unit id, channel (w/ peak indication), and location to lower right
        fig = plt.gcf()
        fig.set_size_inches(10, 4)
        plt.tight_layout()
        # ax = plt.gca()
        # ax.set_aspect(1./ax.get_data_ratio())
        
    # def plot_unit_summary(self):
        
    def _bin_spikes(self, bin_size_s=0.0005):
        """"""
        # add time windowing
        bin_mult = 1000 # make this dynamic
        max_time = np.ceil(max(self.spike_times)*bin_mult)/bin_mult
        times = np.linspace(0, max_time, num=int(max_time/bin_size_s))
        bin_idx = np.digitize(self.spike_times, times)
        amplitudes = np.full((times.size,), 0, dtype=np.float64)
        np.add.at(amplitudes, bin_idx.tolist(), self.spike_amplitudes)
        counts = np.histogram(bin_idx, bins=np.arange(times.size))[0]
        counts = np.concatenate((np.array([1]), counts))
        amplitudes[counts > 0] = amplitudes[counts > 0]/counts[counts > 0]
        return times, counts, amplitudes
        