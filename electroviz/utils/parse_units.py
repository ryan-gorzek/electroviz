
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

from os import listdir
import re

def read_summaries(
        summary_path, 
    ):
    """"""

    file_names = listdir(summary_path)
    probes = [[], [], [], []]
    for f in file_names:
        probe, unit = f.split("_")
        probe_num = int(re.findall("\d+", probe)[0])
        unit_num = int(re.findall("\d+", unit)[0])
        probes[probe_num].append(unit_num)
    return [unit_list for unit_list in probes if len(unit_list) > 0]

