# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from btss.utils import read_visual_protocol, parse_riglog

class bTsStim:
    """
    Interface for Behavior Tasks, Sensory Stimulation (bTsS) experiment databases.
    bTsS is actively developed and maintained by Joao Couto (http://github.com/jcouto).
    """

    def __init__(
            self,
            btss_path=os.getcwd(), 
        ):
        """

        """

        # Load bTsS vislog (stimulus identity) and riglog (timestamp) files.
        assert os.path.exists(bTsS_path), "Could not find the specified path to bTsS data."
        # Locate and read the vislog file
        vislog_file = glob.glob(bTsS_path + "" + "*.vislog")
        assert len(vislog_file) == 1, "The **.vislog file could not be read properly, check that it exists in the path without conflicts.")

        # Locate and read the riglog file
        riglog_file = glob.glob(bTsS_path + "" + "*.riglog")
        assert len(riglog_file) == 1, "The **.riglog file could not be read properly, check that it exists in the path without conflicts.")

    # def get_stim_df():
