# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

class Unit:
    '''
    docstring
    '''
    
    def __init__(self, unit_df, unit_probe_id):
        print('Unit')
        self.id = unit_df.index[0]
        self.info_df = unit_df.loc[[self.id], 
            ["peak_channel_id", "local_index", "cluster_id"]]
        self.info_df.insert(0, "probe_id", unit_probe_id)
        self.info_df.rename(columns={"local_index":"probe_channel_number"})
        self.quality_df = unit_df.loc[[self.id], 
            ["quality", "firing_rate", "presence_ratio", "max_drift",
             "cumulative_drift", "silhouette_score", "isi_violations", 
             "amplitude_cutoff", "isolation_distance", "l_ratio", "d_prime", 
             "nn_hit_rate", "nn_miss_rate", ]]
        self.stats_df = unit_df.loc[[self.id], 
            ["firing_rate", "waveform_duration", "spread", "waveform_halfwidth",
             "snr", "PT_ratio", "repolarization_slope", "recovery_slope", 
             "velocity_above", "velocity_below"]]
        self.stats_df.rename(columns={"spread":"waveform_spread"})
        self.spike_times = unit_df.at[self.id, "spike_times"]
        self.spike_amplitudes = unit_df.at[self.id, "spike_amplitudes"]
        self.mean_waveform = unit_df.at[self.id, "waveform_mean"]

    # def plot_channel_waveforms(self):
        
    # def plot_mean_waveform(self):
        
    # def plot_spike_raster(self):
        
    # def plot_unit_summary(self):
        