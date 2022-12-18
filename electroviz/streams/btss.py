# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import pandas as pd

class bTsS:
    """
    Interface for Behavior Tasks, Sensory Stimulation (bTsS) experiment databases.
    bTsS is actively developed and maintained by Joao Couto (http://github.com/jcouto).
    """


    def __init__(
            self, 
            btss_riglog, 
            btss_visprot, 
        ):
        """
        Parse bTsS directory.
        """

        # Store the rig log.
        self.riglog = btss_riglog




class bTsSVStim(bTsS):
    """

    """


    def __init__(
            self, 
            btss_riglog, 
            btss_visprot, 
        ):
        """"""

        super().__init__(
                    btss_riglog, 
                    btss_visprot)

        self.visprot = btss_visprot
        self.vstim_params = self._get_vstim_params()
        self.vstim_df = self._get_vstim_df()

    def _get_vstim_df(
            self, 
        ):
        """"""
        
        # Get the vstim dataframe from the rig log.
        vstim_df = self.riglog[0]["vstim"]
        # Drop unnecessary columns.
        vstim_df.drop(["code"], axis=1, inplace=True)
        # Transpose to match other time-series data.
        vstim_df_T = vstim_df.T
        return vstim_df_T


    def _get_vstim_params(
            self, 
        ):
        """"""

        # Remove unnecessary parameters from the visprot.
        visprot_dict = self.visprot[0]
        del visprot_dict["stim_type"], visprot_dict["indicator_mode"], visprot_dict["name"]
        return visprot_dict

        
