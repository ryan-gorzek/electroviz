# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from electroviz.io.reader import read_bTsS
from electroviz.streams.vstim import VStim


class bTsS:
    """
    Interface for Behavior Tasks, Sensory Stimulation (bTsS) experiment databases.
    bTsS is actively developed and maintained by Joao Couto (http://github.com/jcouto).
    """


    def __new__(
            self, 
            btss_path, 
            index=0, 
        ):
        """"""

        # Read the visual protocol and rig log files.
        visprot, riglog = read_bTsS(btss_path)
        #
        btss = []
        # 
        vstim = VStim(visprot, riglog[0]["vstim"], index=index)
        btss.append(vstim)
        return btss
