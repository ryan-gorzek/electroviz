# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_Kilosort
from electroviz.streams.spikes import Spikes

class Kilosort:
    """
    
    """

    def __new__(
            self, 
            kilosort_path, 
            total_imec_samples, 
        ):

        # Read spike clusters and times from Kilosort output.
        kilosort_dict = read_Kilosort(kilosort_path)
        kilosort = []
        kilosort.append(Spikes(total_imec_samples, 
                               kilosort_dict))
        return kilosort
