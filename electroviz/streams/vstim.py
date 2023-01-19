
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd

class VStim:
    """

    """


    def __init__(
            self, 
            btss_visprot, 
            btss_vstim, 
            index=0, 
        ):
        """"""

        self.visprot = btss_visprot
        self._events_all = btss_vstim.drop(["code"], axis=1)
        events = []
        prev_row = None
        for row in self._events_all.itertuples(index=True):
            if row[0] == 0:
                events.append(row[1:])
            elif row[-1] != prev_row:
                events.append(row[1:])
            prev_row = row[-1]
        self.events = pd.DataFrame(events, columns=self._events_all.columns)
        self.index = index

