# MIT License
# Copyright (c) 2022 Ryan Gorzek
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
        ):
        """"""

        self.visprot = btss_visprot
        self.vstim = btss_vstim.drop(["code"], axis=1, inplace=True)
