# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from typing import NamedTuple

class Event(NamedTuple):
    """

    """
    
    index: int = None
    sample_onset: int = None
    sample_offset: int = None
    sample_duration: int = None
    time_onset: float  = None
    time_offset: float = None
    time_duration: float = None
    digital_value: int = None
