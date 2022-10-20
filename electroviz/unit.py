# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt

class Unit:
    '''
    docstring
    '''
    
    def __init__(self, unit_df, unit_probe_id, unit_location):
        """"""
        print('Unit')
        self.id = unit_df.index[0]
        self.info_df = unit_df.loc[[self.id], 
            ["peak_channel_id", "local_index", "cluster_id"]]
        self.info_df.insert(0, "probe_id", unit_probe_id)
        self.info_df["location"] = unit_location
        self.info_df.rename(columns={"local_index":"probe_channel_number"}, inplace=True)
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
        self.mean_waveforms = unit_df.at[self.id, "waveform_mean"]
        
    def plot_mean_waveform(self, channel="peak", color="k"):
        """"""
        time_series = np.linspace(0, 2.7, num=82)
        if channel == "peak":
            plot_channel = self.info_df.at[self.id, "probe_channel_number"]
        plt.plot(time_series, self.mean_waveforms[plot_channel, :], color=color)
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
        
    # def plot_spike_raster(self):
        
    # def plot_unit_summary(self):
        