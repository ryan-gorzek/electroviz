# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
from scipy import sparse

def align_sync(
        object_1, 
        object_2, 
    ):
    """
    Align two objects based on their sync signals.
    """

    # Get sample onsets, offsets, and durations from each object's sync signal.
    event_names = ["sample_onset", "sample_offset"]
    sample_events_1 = object_1.sync.digital_df.loc[event_names].to_numpy().T.astype(int)
    sample_events_2 = object_2.sync.digital_df.loc[event_names].to_numpy().T.astype(int)
    # Preallocate index arrays for dropping samples.
    sample_drop_1 = []
    sample_drop_2 = []
    for event_idx, ((on_1, off_1), (on_2, off_2)) in enumerate(zip(sample_events_1, sample_events_2)):

        # Drop an initial sample to align the onset of the first pulse.
        if event_idx == 0:
            if on_1 > on_2:
                [sample_drop_1.append(idx) for idx in range(int(on_1 - on_2))]
            elif on_1 < on_2:
                [sample_drop_2.append(idx) for idx in range(int(on_2 - on_1))]
            # Account for the leftward shift in offset times on the first pulse.
            off_1 += -len(sample_drop_1)
            off_2 += -len(sample_drop_2)
        else:
            # Account for the leftward shift in onset and offset times from all previous pulses.
            on_1 += -len(sample_drop_1)
            off_1 += -len(sample_drop_1)
            on_2 += -len(sample_drop_2)
            off_2 += -len(sample_drop_2)

        # Downsample the pulse (from end) with the larger number of samples if unequal.
        if off_1 > off_2:
            [sample_drop_1.append(idx + 1) for idx in range(off_2, off_1)]
        elif off_1 < off_2:
            [sample_drop_2.append(idx + 1) for idx in range(off_1, off_2)]

    # If one of the signals is longer, drop samples from the end to match.
    tot_1, tot_2 = object_1.sync.total_samples, object_2.sync.total_samples
    rem_1, rem_2 = tot_1 - len(sample_drop_1), tot_2 - len(sample_drop_2)
    if rem_1 > rem_2:
        num_drop = rem_1 - rem_2
        [sample_drop_1.append(idx) for idx in range(int(tot_1 - num_drop), tot_1)]
    elif rem_1 < rem_2:
        num_drop = rem_2 - rem_1
        [sample_drop_2.append(idx) for idx in range(int(tot_2 - num_drop), tot_2)]

    # Rebuild the sync signals, dropping the indices identified here.
    object_1.sync.rebuild(sample_drop_1)
    object_2.sync.rebuild(sample_drop_2)
    return object_1, object_2
