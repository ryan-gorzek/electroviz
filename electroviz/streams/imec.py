# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np
from scipy import sparse

class Imec:
    """

    """

    def __init__(
            self, 
            imec_metadata, 
            imec_binary, 
            kilosort_array, 
        ):

        # Get some basic data and parameters for easy access.
        self.imec_metadata = imec_metadata
        self.imec_binary = imec_binary
        self.sampling_rate = float(self.imec_metadata["imSampRate"])
        self.total_time = float(self.imec_metadata["fileTimeSecs"])
        self.total_samples = int(self.sampling_rate * self.total_time)




# class ImecProbe(Imec):
#     """

#     """

#     def __init__(
#             self, 
#             imec_metadata, 
#             imec_binary, 
#             kilosort_array, 
#         ):




class ImecSpikes(Imec):
    """
    
    """

    def __init__(
            self, 
            imec_metadata, 
            imec_binary, 
            kilosort_array, 
        ):

        super().__init__(imec_metadata, 
                         imec_binary, 
                         kilosort_array)

        # Get Kilosort data from columns of kilosort_array.
        (spike_clusters, spike_times) = np.hsplit(kilosort_array.flatten(), 2)
        # Get the total number of units identified by Kilosort (index starts at 0).
        self.total_units = int(np.max(spike_clusters) + 1)
        # Build (sparse) spike times matrix.
        self.spike_times = self._build_spike_times_matrix(spike_clusters, spike_times)

    def _build_spike_times_matrix(
            self, 
            spike_clusters, 
            spike_times, 
        ):
        """"""

        # Create a units-by-samples scipy sparse coordinate matrix to store spike times.
        full_shape = (self.total_units, self.total_samples)
        row_idx = spike_clusters
        col_idx = spike_times
        data = np.ones((spike_times.size,))
        spike_times_matrix = sparse.coo_matrix((data, (row_idx, col_idx)), shape=full_shape)
        return spike_times_matrix


# class ImecSync(Imec):
#     """

#     """

#     def __init__(
#             self, 
#             imec_metadata, 
#             imec_binary, 
#             kilosort_array, 
#         ):
