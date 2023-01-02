# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
from warnings import warn
import glob
from pathlib import Path
import numpy as np
from btss.utils import read_visual_protocol, parse_riglog


def parse_experiment_dir(
        experiment_path, 
        SGLX_name, 
        bTsS_names, 
    ):
    """"""

    # Check if specified path is a directory [or .electroviz parameters file].
    assert os.path.exists(experiment_path), "The specified path is not valid."
    ephys_subdir, protocol_subdir = None, None
    if os.path.isdir(experiment_path):
        for subdir in os.listdir(experiment_path):
            # Check for ephys data directory.
            if SGLX_name in subdir:
                SGLX_subdir = subdir
            # Check for protocol directory.
            for protocol in bTsS_names:
                if protocol in subdir:
                    bTsS_subdir = protocol
        # Need ephys and/or protocol data to do anything.
        if (SGLX_subdir is None) & (bTsS_subdir is None):
            raise Exception("Could not find SpikeGLX or bTsS data.")
    return SGLX_subdir, bTsS_subdir


def parse_SGLX_dir(
        SGLX_path,
    ):
    """Find NIDAQ and Imec binary and metadata files within SpikeGLX subdirectory and return their paths."""
    imec_path = []
    # Move through SpikeGLX directory, expecting NI-DAQ data at the top and subfolders with Imec data.
    for topdir, _, files in os.walk(SGLX_path):
        SGLX_files = [f if (f.endswith(".bin") | f.endswith(".meta")) else "" for f in files]
        # Check whether this directory has NIDAQ binary and metadata files.
        nidaq_count = sum(1 for f in SGLX_files if (f.endswith(".nidq.bin") | f.endswith(".nidq.meta")))
        if nidaq_count == 2:
            nidaq_path = topdir
        # Check whether this directory has Imec binary and metadata files.
        imec_count = sum(1 for f in SGLX_files if ((".ap" in f) & (".bin" in f)) | 
                                                    ((".ap" in f) & (".meta" in f)))
        if imec_count == 2:
            imec_path.append(topdir)
    # Just return a string for the Imec data if there's only one folder.
    if len(imec_path) == 1: imec_path = imec_path[0]
    return nidaq_path, imec_path


def read_Kilosort(
        kilosort_path, 
    ):
    """"""

    # Read Kilosort output files into numpy array.
    kilosort_names = ["spike_clusters.npy", "spike_times.npy"]
    kilosort_array = []
    for name in kilosort_names:
        kilosort_array.append(np.load(kilosort_path + "/" + name))
    kilosort_array = np.array(kilosort_array).squeeze().T
    (spike_clusters, spike_times) = np.hsplit(kilosort_array.flatten(), 2)
    return spike_clusters, spike_times


def read_Imec(
        imec_path, 
    ):
    """Read Imec metadata and binary files, as well as Kilosort output, from path."""

    # Read the metadata using SpikeGLX datafile tools.
    metadata_path = glob.glob(imec_path + "/*.ap.meta")[0]
    imec_metadata = readMeta(metadata_path)
    # Read the binary file using SpikeGLX datafile tools.
    binary_path = glob.glob(imec_path + "/*.ap.bin")[0]
    imec_binary = makeMemMapRaw(binary_path, imec_metadata)
    return imec_metadata, imec_binary


def read_NIDAQ(
        nidaq_path, 
    ):
    """Read NI-DAQ metadata and binary files from path."""

    # Read the metadata using SpikeGLX datafile tools.
    metadata_path = glob.glob(nidaq_path + "/*.nidq.meta")[0]
    nidaq_metadata = readMeta(metadata_path)
    # Read the binary file using SpikeGLX datafile tools.
    binary_path = glob.glob(nidaq_path + "/*.nidq.bin")[0]
    nidaq_binary = makeMemMapRaw(binary_path, nidaq_metadata)
    return nidaq_metadata, nidaq_binary


def read_bTsS(
        btss_path, 
    ):
    """"""

    # Read the .visprot file using bTsS tools.
    visprot_path = glob.glob(btss_path + "/" + "*.visprot")[0]
    btss_visprot = read_visual_protocol(visprot_path)
    # Read the .riglog file using bTsS tools.
    riglog_path = glob.glob(btss_path + "/" + "*.riglog")[0]
    btss_riglog = parse_riglog(riglog_path)
    return btss_visprot, btss_riglog


# Methods for reading binary (analog or digital) and metadata files generated by SpikeGLX.
# These tools are found on the SpikeGLX website (https://billkarsh.github.io/SpikeGLX/).
# SpikeGLX is actively developed and maintained by Bill Karsh (https://github.com/billkarsh).

def readMeta(metaFile):
    metaDict = {}
    with Path(metaFile).open() as f:
        mdatList = f.read().splitlines()
        # convert the list entries into key value pairs
        for m in mdatList:
            csList = m.split(sep='=')
            if csList[0][0] == '~':
                currKey = csList[0][1:len(csList[0])]
            else:
                currKey = csList[0]
            metaDict.update({currKey: csList[1]})
    return(metaDict)


def makeMemMapRaw(binFile, meta):
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    rawData = np.memmap(binFile, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)
