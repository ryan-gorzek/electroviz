
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from typing import NamedTuple

class Event(NamedTuple):
    """

    """


    stim_indices: tuple = None
    index: int = None
    sample_onset: int = None
    sample_offset: int = None
    sample_duration: int = None
    digital_value: int = None
    contrast: int = None
    posx: int = None
    posy: int = None
    ori: int = None
    sf: int = None
    phase: int = None
    tf: int = None
    itrial: int = None
    istim: int = None

