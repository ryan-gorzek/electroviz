
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
from scipy import sparse
import copy

def align_ap_sync(
        nidaq, 
        imec_ap, 
        kilosort, 
    ):
    """"""

    syncs = [nidaq[0], *imec_ap]
    # Get sample onsets, offsets, and durations from each object's sync signal.
    event_names = ["sample_onset", "sample_offset"]
    events = []
    for sync in syncs:
        events.append(sync.events[event_names].to_numpy().astype(int))
    # Preallocate index array for dropping samples.
    drops = [[] for s in syncs]
    for event_idx, ons_offs in enumerate(zip(*events)):
        ons, ons_th = np.concatenate(ons_offs)[0::2], np.concatenate(ons_offs)[0::2]
        offs, offs_th = np.concatenate(ons_offs)[1::2], np.concatenate(ons_offs)[1::2]
        # Drop an initial sample to align the onset of the first pulse.
        if event_idx == 0:
            ons_diff = ons - np.min(ons)
            for sync_idx, on in enumerate(ons_diff):
                [drops[sync_idx].append(idx) for idx in range(int(on)) if on != 0]
            # Account for the leftward shift in offset times on the first pulse.
            offs_th += np.array([-len(d) for d in drops])
        else:
            # Account for the leftward shift in onset and offset times from all previous pulses.
            ons_th += np.array([-len(d) for d in drops])
            offs_th += np.array([-len(d) for d in drops])

        # Downsample the pulse (from end) with the larger number of samples if unequal.
        for sync_idx, (off, off_th) in enumerate(zip(offs, offs_th)):
            diff = off_th - np.min(offs_th)
            idx_range = range(off - diff, off)
            [drops[sync_idx].append(idx + 1) for idx in idx_range if off_th != np.min(offs_th)]

    # If one of the signals is longer, drop samples from the end to match.
    totals = np.array([sync.total_samples for sync in syncs])
    rems = totals - np.array([len(d) for d in drops])
    rems_diff = rems - np.min(rems)
    for sync_idx, (tot, rem) in enumerate(zip(totals, rems_diff)):
        [drops[sync_idx].append(idx) for idx in range(int(tot - rem), tot) if rem != 0]

    # Rebuild the NIDAQ signals, dropping the indices identified here.
    nidaq_ap = copy.deepcopy(nidaq)
    for obj in nidaq_ap:
        obj.drop_and_rebuild(drops[0])
    # Imec.
    for obj, drop in zip(imec_ap, drops[1:]):
        obj.drop_and_rebuild(drop)
    # Kilosort.
    for obj, drop in zip(kilosort, drops[1:]):
        obj.drop_and_rebuild(drop)

    return nidaq_ap, imec_ap, kilosort


def align_lf_sync(
        nidaq, 
        imec_lf, 
    ):
    """"""

    # Align the LFP syncs first.
    syncs = [*imec_lf[::2]]
    # Get sample onsets, offsets, and durations from each object's sync signal.
    event_names = ["sample_onset", "sample_offset"]
    events = []
    for sync in syncs:
        events.append(sync.events[event_names].to_numpy().astype(int))
    # Preallocate index array for dropping samples.
    drops = [[] for s in syncs]
    for event_idx, ons_offs in enumerate(zip(*events)):
        ons, ons_th = np.concatenate(ons_offs)[0::2], np.concatenate(ons_offs)[0::2]
        offs, offs_th = np.concatenate(ons_offs)[1::2], np.concatenate(ons_offs)[1::2]
        # Drop an initial sample to align the onset of the first pulse.
        if event_idx == 0:
            ons_diff = ons - np.min(ons)
            for sync_idx, on in enumerate(ons_diff):
                [drops[sync_idx].append(idx) for idx in range(int(on)) if on != 0]
            # Account for the leftward shift in offset times on the first pulse.
            offs_th += np.array([-len(d) for d in drops])
        else:
            # Account for the leftward shift in onset and offset times from all previous pulses.
            ons_th += np.array([-len(d) for d in drops])
            offs_th += np.array([-len(d) for d in drops])

        # Downsample the pulse (from end) with the larger number of samples if unequal.
        for sync_idx, (off, off_th) in enumerate(zip(offs, offs_th)):
            diff = off_th - np.min(offs_th)
            idx_range = range(off - diff, off)
            [drops[sync_idx].append(idx + 1) for idx in idx_range if off_th != np.min(offs_th)]

    # If one of the signals is longer, drop samples from the end to match.
    totals = np.array([sync.total_samples for sync in syncs])
    rems = totals - np.array([len(d) for d in drops])
    rems_diff = rems - np.min(rems)
    for sync_idx, (tot, rem) in enumerate(zip(totals, rems_diff)):
        [drops[sync_idx].append(idx) for idx in range(int(tot - rem), tot) if rem != 0]
    
    # Imec.
    for obj, drop in zip(imec_lf, drops):
        obj.drop_and_rebuild(drop)

    # Then, match the NIDAQ signals to the LFP syncs.
    imec_events = [imec_lf[0].events[event_names].to_numpy().astype(int)]
    nidaq_events, new_events = [], []
    for ni in nidaq:
        nidaq_events.append(ni.events[event_names].to_numpy().astype(int))
        new_events.append(np.empty(ni.events[event_names].to_numpy().astype(int).shape))
    pc_sets = nidaq_events[1].flatten()
    photo_sets = nidaq_events[2].flatten()
    for event_idx, ons_offs in enumerate(zip(imec_events[0], nidaq_events[0])):
        imec_onoff = ons_offs[0]
        sync_on, sync_off = ons_offs[1]
        # Set NIDAQ sync events to Imec sync events.
        new_events[0][event_idx, :] = imec_onoff
        # Map PC Clock to LFP sync.
        pc_log = np.logical_and(pc_sets >= sync_on, pc_sets <= sync_off)
        if any(pc_log):
            pc_in = pc_sets[pc_log]
            for pc_set in pc_in:
                pc_loc = int(((pc_set - sync_on) / (sync_off - sync_on)) * np.diff(imec_onoff)[0])
                row, col = np.where(nidaq_events[1] == pc_set)
                new_events[1][row, col] = int(pc_loc + imec_onoff[0])
        # Map Photodiode to LFP sync.
        photo_log = np.logical_and(photo_sets >= sync_on, photo_sets <= sync_off)
        if any(photo_log):
            photo_in = photo_sets[photo_log]
            for photo_set in photo_in:
                photo_loc = int(((photo_set - sync_on) / (sync_off - sync_on)) * np.diff(imec_onoff)[0])
                row, col = np.where(nidaq_events[2] == photo_set)
                new_events[2][row, col] = int(photo_loc + imec_onoff[0])
    nidaq_lf = copy.deepcopy(nidaq)
    for ni_lf, new_ev in zip(nidaq_lf, new_events):
        ni_lf.events[event_names] = new_ev.astype(int)

    return nidaq_lf, imec_lf
