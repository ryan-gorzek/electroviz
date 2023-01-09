# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
from scipy import sparse

def align_sync(
        nidaq, 
        imec,
        kilosort,  
    ):
    """
    
    """

    syncs = [nidaq[0], *imec[0:2]]
    # Get sample onsets, offsets, and durations from each object's sync signal.
    event_names = ["sample_onset", "sample_offset"]
    events = []
    for sync in syncs:
        events.append(sync.events[event_names].to_numpy().astype(int))
    # Preallocate index array for dropping samples.
    drops = [[] for s in syncs]
    for event_idx, ons_offs in enumerate(zip(*events)):
        ons = np.concatenate(ons_offs)[0::2]
        offs = np.concatenate(ons_offs)[1::2]
        ons_diff = ons - np.min(ons)
        # Drop an initial sample to align the onset of the first pulse.
        if event_idx == 0:
            for sync_idx, on in enumerate(ons_diff):
                [drops[sync_idx].append(idx) for idx in range(int(on)) if on != 0]
            # Account for the leftward shift in offset times on the first pulse.
            offs += np.array([-len(d) for d in drops])
        else:
            # Account for the leftward shift in onset and offset times from all previous pulses.
            ons += np.array([-len(d) for d in drops])
            offs += np.array([-len(d) for d in drops])

        # Downsample the pulse (from end) with the larger number of samples if unequal.
        for sync_idx, off in enumerate(offs):
            (drops[sync_idx].append(idx + 1) for idx in range(int(np.min(offs)), off) if off != np.min(offs))

    # If one of the signals is longer, drop samples from the end to match.
    totals = np.array([sync.total_samples for sync in syncs])
    rems = totals - np.array([len(d) for d in drops])
    rems_diff = rems - np.min(rems)
    for sync_idx, (tot, rem) in enumerate(zip(totals, rems_diff)):
        (drops[sync_idx].append(idx) for idx in range(int(tot - rem), tot) if rem != 0)

    # Rebuild the NIDAQ signals, dropping the indices identified here.
    for obj in nidaq:
        obj.drop_and_rebuild(drops[0])
    # Imec.
    for obj, drop in zip(imec, drops[1:]):
        obj.drop_and_rebuild(drop)
    # Kilosort.
    for obj, drop in zip(kilosort, drops[1:]):
        obj.drop_and_rebuild(drop)

    return nidaq, imec, kilosort
