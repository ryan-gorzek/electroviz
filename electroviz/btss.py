# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
import glob
from btss.utils import read_visual_protocol, parse_riglog

class bTsS:
    """
    Interface for Behavior Tasks, Sensory Stimulation (bTsS) experiment databases.
    bTsS is actively developed and maintained by Joao Couto (http://github.com/jcouto).
    """


    def __init__(
            self,
            exp_path=os.getcwd(), 
            protocol_names=["contra_random_squares", "ipsi_random_squares"]
        ):
        """
        Parse bTsS directory.
        """

        # Load bTsS vislog (stimulus identity) and riglog (timestamp) files.
        assert os.path.exists(exp_path), "Could not find the specified path to bTsS data."
        # Check for folder containing one of the protocol names.
        protocol_subdir = None
        for subdir in os.listdir(exp_path):
            if subdir in protocol_names:
                protocol_subdir = subdir
        # assert protocol_name is not None, 
        #     "bTsS folder exists, but could not find protocol folder. Check that protocol_names is correct."

        # Get the protocol name as a string and append it to the bTsS path.
        protocol_path = exp_path + "/" + protocol_subdir
        # Read the visprot file from the protocol folder.
        visprot_file = glob.glob(protocol_path + "/" + "*.visprot")
        assert len(visprot_file) == 1, "The **.visprot file could not be read properly, check that it exists in the path without conflicts."
        self.visprot = read_visual_protocol(visprot_file[0])
        # Read the riglog file from the protocol folder.
        riglog_file = glob.glob(protocol_path + "/" + "*.riglog")
        assert len(riglog_file) == 1, "The **.riglog file could not be read properly, check that it exists in the path without conflicts."
        self.riglog = parse_riglog(riglog_file[0])
