# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

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

        print(kilosort_array.shape)

    # def _build_times_matrix(
    #         self, 
    #         spike_times, 
    #     ):

# class ImecSync(Imec):
#     """

#     """

#     def __init__(
#             self, 
#             imec_metadata, 
#             imec_binary, 
#             kilosort_array, 
#         ):
