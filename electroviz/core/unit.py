# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt
from math import floor, log10
# from scipy.stats import zscore

class Unit:
    """
    docstring
    """
    
    #### Special Methods ####
    
    def __init__(
            self, 
            unit_df, 
            unit_probe_id, 
            unit_local_index, 
            unit_location, 
        ):
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
        
    #### Plotting Methods ####
        
    def plot_mean_waveform(
            self, 
            channel="peak", 
            color="k",
        ):
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
        
    def plot_spike_raster(
            self, 
            bin_size_s=0.0005, 
            time_window_s=(None, None), 
            raster_type="amplitudes",
        ):
        """"""
        if raster_type == "amplitudes":
            _, _, amplitudes = self._bin_spikes(bin_size_s=bin_size_s, time_window_s=time_window_s)
            plt.imshow(np.expand_dims(amplitudes, axis=0), 
                       aspect=0.2*amplitudes.size, 
                       cmap="gray_r",
                       clim=[0, np.nanmean(amplitudes)])
        # elif raster_type == "counts":
        # elif raster_type == "binary":
        plt.xlabel("Time (s)")
        plt.yticks([])
        # set x limits and ticks dynamically
        # add unit id, channel (w/ peak indication), and location to lower right
        fig = plt.gcf()
        fig.set_size_inches(10, 4)
        plt.tight_layout()
        ax = plt.gca()
        return ax
    
    def plot_aligned_response(
            self, 
            align_to=None, 
            rel_window_s=(None, None), 
            bin_size_s=0.0005, 
            raster_type="amplitudes", 
        ):
        """"""
        if align_to != None:
            if raster_type == "amplitudes":
                amplitudes = []
                for start, stop in align_to.info_df[["start_time", "stop_time"]].values:
                    time_window_s = (start + rel_window_s[0], start + rel_window_s[1])
                    curr_times, _, curr_amps = self._bin_spikes(bin_size_s=bin_size_s, time_window_s=time_window_s)
                    amplitudes.append(curr_amps)
                amplitudes = np.array(amplitudes, dtype=object)
                shape_ratio = amplitudes.shape[1] / amplitudes.shape[0]
                plt.imshow(amplitudes.astype(np.float64), 
                           aspect=(0.4*shape_ratio), 
                           cmap="gray_r", 
                           clim=[0, np.nanmean(amplitudes.astype(np.float64), axis=(0,1))])
                plt.xlabel("Time to Onset (s)\n")
                plt.yticks([])
                plt.xticks(ticks=np.linspace(0, amplitudes.shape[1], num=11), 
                           labels=np.around(np.linspace(*rel_window_s, num=11), decimals=1))
                # add unit id, channel (w/ peak indication), and location to lower right
                plt.tight_layout()
                fig = plt.gcf()
                fig.set_size_inches(10, 5)
                plt.tight_layout()
                ax = plt.gca()
        return ax
    
    def plot_averaged_response(
            self, 
            align_to=None, 
            rel_window_s=(None, None), 
            bin_size_s=0.0005, 
            bound_type="STD",
        ):
        """"""
        if align_to != None:
            amplitudes = []
            for start, stop in align_to.info_df[["start_time", "stop_time"]].values:
                time_window_s = (start + rel_window_s[0], start + rel_window_s[1])
                curr_times, _, curr_amps = self._bin_spikes(bin_size_s=bin_size_s, time_window_s=time_window_s)
                amplitudes.append(curr_amps)
            amplitudes = np.array(amplitudes)
            response_avg = np.mean(amplitudes, axis=0)
            if bound_type == "STD":
                bound_upper = response_avg + np.std(amplitudes, axis=0)
                bound_lower = response_avg - np.std(amplitudes, axis=0)
            elif bound_type == "SEM":
                bound_upper = response_avg + np.std(amplitudes, axis=0)/np.sqrt(amplitudes.shape[0])
                bound_lower = response_avg - np.std(amplitudes, axis=0)/np.sqrt(amplitudes.shape[0])
            elif bound_type == "range":
                bound_upper = np.max(amplitudes, axis=0)
                bound_lower = np.min(amplitudes, axis=0)
            plt.plot(np.arange(0, amplitudes.shape[1], 1), 
                     response_avg, 
                     color="r")
            plt.fill_between(np.arange(0, amplitudes.shape[1], 1), 
                             bound_lower, 
                             bound_upper, 
                             color=(1,0,0,0.1))
            plt.xlabel("Time to Onset (s)\n")
            plt.yticks([])
            plt.xticks(ticks=np.linspace(0, amplitudes.shape[1], num=11), 
                       labels=np.around(np.linspace(*rel_window_s, num=11), decimals=1))
            # add unit id, channel (w/ peak indication), and location to lower right
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            plt.tight_layout()
            ax = plt.gca()
        return ax
        
    # def plot_unit_summary(self):
        
    #### Quantification Methods ####
    
    
        
    #### Private Methods ####
        
    def _bin_spikes(
            self, 
            bin_size_s=0.0005, 
            time_window_s=(None, None), 
        ):
        """"""
        time_start = 0 if time_window_s[1] == None else time_window_s[0]
        time_stop = max(self.spike_times) if time_window_s[1] == None else time_window_s[1]
        num_decimals = -floor(log10(abs(bin_size_s)))
        bin_mult = 10**(num_decimals - 1)
        min_time = np.floor(time_start*bin_mult)/bin_mult
        max_time = np.ceil(time_stop*bin_mult)/bin_mult
        times = np.linspace(min_time, max_time, 
                            num=int(np.around(max_time - min_time, decimals=num_decimals)/bin_size_s))
        spike_times = self.spike_times[(self.spike_times >= min_time) & 
                                       (self.spike_times <= max_time)]
        bin_idx = np.digitize(spike_times, times)
        amplitudes = np.full((times.size,), 0, dtype=np.float64)
        spike_amplitudes = self.spike_amplitudes[(self.spike_times >= min_time) & 
                                                 (self.spike_times <= max_time)]
        np.add.at(amplitudes, bin_idx.tolist(), spike_amplitudes)
        counts = np.histogram(bin_idx, bins=np.arange(times.size))[0]
        counts = np.concatenate((np.array([1]), counts))
        amplitudes[counts > 0] = amplitudes[counts > 0]/counts[counts > 0]
        return times, counts, amplitudes
        