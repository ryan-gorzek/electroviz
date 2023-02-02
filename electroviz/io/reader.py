
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
from warnings import warn
import glob
from pathlib import Path
import numpy as np
from btss.utils import read_visual_protocol, parse_riglog

def read_config(
        experiment_path, 
    ):
    """"""
    
    assert os.path.exists(experiment_path), "The specified path is not valid."
    config_filepath = glob.glob(experiment_path + "/*.electroviz.config")[0]
    assert os.path.exists(config_filepath), "Can't find electroviz config file."
    SGLX_name = np.loadtxt(config_filepath, max_rows=1, dtype=str)[1]
    NIDAQ_gates = list(np.loadtxt(config_filepath, skiprows=1, max_rows=1, dtype=str)[1:])
    bTsS_names = list(np.loadtxt(config_filepath, skiprows=2, max_rows=1, dtype=str)[1:])
    return SGLX_name, NIDAQ_gates, bTsS_names

def parse_experiment_dir(
        experiment_path, 
        SGLX_name, 
        bTsS_names, 
    ):
    """"""

    # Check if specified path is a directory [or .electroviz parameters file].
    assert os.path.exists(experiment_path), "The specified path is not valid."
    SGLX_subdir, bTsS_subdir = None, []
    if os.path.isdir(experiment_path):
        for subdir in os.listdir(experiment_path):
            # Check for ephys data directory.
            if SGLX_name in subdir:
                SGLX_subdir = subdir
            # Check for protocol directory.
            for protocol in bTsS_names:
                if protocol in subdir:
                    bTsS_subdir.append(protocol)
        # Need ephys and/or protocol data to do anything.
        if (SGLX_subdir is None) & (bTsS_subdir is []):
            raise Exception("Could not find SpikeGLX or bTsS data.")
        else:
            def bTsS_loc(name):
                return bTsS_names.index(name)
            bTsS_subdir.sort(key=bTsS_loc)
    return SGLX_subdir, bTsS_subdir


def parse_SGLX_dir(
        SGLX_path,
    ):
    """"""

    imec_paths = []
    # Move through SpikeGLX directory, expecting NI-DAQ data at the top and subfolders with Imec data.
    for topdir, _, files in os.walk(SGLX_path):
        SGLX_files = [f if (f.endswith(".bin") | f.endswith(".meta")) else "" for f in files]
        # Check whether this directory has NIDAQ binary and metadata files.
        nidaq_count = sum(1 for f in SGLX_files if (f.endswith(".nidq.bin") | f.endswith(".nidq.meta")))
        if nidaq_count > 0:
            nidaq_path = topdir
        # Check whether this directory has Imec binary and metadata files.
        imec_count = sum(1 for f in SGLX_files if ((".ap" in f) & (".bin" in f)) | 
                                                    ((".ap" in f) & (".meta" in f)))
        if imec_count == 2:
            imec_paths.append(topdir)
    return nidaq_path, imec_paths


def read_Kilosort(
        kilosort_paths, 
    ):
    """"""

    kilosort_dicts = []
    for path in kilosort_paths:
        # Read Kilosort spike data into numpy arrays.
        kilosort_names = ["cluster_info.tsv", "spike_times.npy", "spike_clusters.npy"]
        kilosort_dict = {}
        for fname in kilosort_names:
            key, ext = Path(fname).stem, Path(fname).suffix
            if "cluster_info" in key:
                kilosort_dict["cluster_id"] = np.loadtxt(path + "/" + fname, dtype=str, skiprows=1, usecols=0)
                kilosort_dict["cluster_quality"] = np.loadtxt(path + "/" + fname, dtype=str, skiprows=1, usecols=3)
                kilosort_dict["peak_channel"] = np.loadtxt(path + "/" + fname, dtype=str, skiprows=1, usecols=5)
                kilosort_dict["depth"] = np.loadtxt(path + "/" + fname, dtype=str, skiprows=1, usecols=6)
            else:
                kilosort_dict[key] = np.load(path + "/" + fname)
        kilosort_dicts.append(kilosort_dict)
    return kilosort_dicts


def read_Imec(
        imec_paths, 
    ):
    """"""

    ap_metadata, ap_binary, lf_metadata, lf_binary = [], [], [], []
    for path in imec_paths:
        # Read the ap metadata using SpikeGLX datafile tools.
        ap_metadata_path = glob.glob(path + "/*.ap.meta")[0]
        ap_meta = readMeta(ap_metadata_path)
        ap_metadata.append(ap_meta)
        # Read the ap binary file using SpikeGLX datafile tools.
        ap_binary_path = glob.glob(path + "/*.ap.bin")[0]
        ap_binary.append(makeMemMapRaw(ap_binary_path, ap_meta))
        # Read the lf metadata using SpikeGLX datafile tools.
        lf_metadata_path = glob.glob(path + "/*.lf.meta")[0]
        lf_meta = readMeta(lf_metadata_path)
        lf_metadata.append(lf_meta)
        # Read the lf binary file using SpikeGLX datafile tools.
        lf_binary_path = glob.glob(path + "/*.lf.bin")[0]
        lf_binary.append(makeMemMapRaw(lf_binary_path, lf_meta))
    return ap_metadata, ap_binary, lf_metadata, lf_binary, imec_paths


def read_NIDAQ(
        nidaq_path, 
        nidaq_gates, 
    ):
    """"""

    metadata_paths = glob.glob(nidaq_path + "/*.nidq.meta")
    binary_paths = glob.glob(nidaq_path + "/*.nidq.bin")
    meta_paths, bnry_paths = [], []
    for gate in nidaq_gates:
        gate_id = "ephys_g{0}_t0".format(gate)
        for meta, bnry in zip(metadata_paths, binary_paths):
            if (gate_id in meta) and (gate_id in bnry):
                meta_paths.append(meta)
                bnry_paths.append(bnry)
    nidaq_metadata, nidaq_binary = [], []
    for meta, bnry in zip(meta_paths, bnry_paths):
        nidaq_metadata.append(readMeta(meta))
        nidaq_binary.append(makeMemMapRaw(bnry, nidaq_metadata[-1]))
    return nidaq_binary, nidaq_metadata


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

