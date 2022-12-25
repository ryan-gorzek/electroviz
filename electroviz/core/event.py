# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from collections import namedtuple

class Event:
    """

    """

    
    def __new__(
            self, 
            event_idx, 
            event_data, 
        ):
        """"""

        event = namedtuple("Event", ["index"] + list(event_data.index))
        
        return event()
