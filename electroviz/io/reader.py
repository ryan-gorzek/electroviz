# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
from warnings import warn
import glob
from pathlib import Path
import numpy as np
from electroviz.streams.nidaq import NIDAQSync, NIDAQDigital
from electroviz.streams.imec import ImecSpikes, ImecSync
from electroviz.core.population import Population
from electroviz.streams.btss import bTsSVStim
from electroviz.core.stimulus import SparseNoise
from btss.utils import read_visual_protocol, parse_riglog

class Reader:
    """

    """


    def __init__(
            self, 
            path, 
        ):
        """

        """

        # Put this stuff in a config file later.
        ephys_name = "ephys"
        protocol_names = ["ipsi_random_squares"]
        self.nidaq_digital = dict({
            # "camera" :     {"line_num":5}, 
            "pc_clock" :   {"line_num" : 4}, 
            "photodiode" : {"line_num" : 1}, 
                                   })

        # Check if specified path is a directory [or .electroviz parameters file].
        assert os.path.exists(path), "The specified path is not valid."
        ephys_subdir = None
        protocol_subdir = None
        if os.path.isdir(path):
            for subdir in os.listdir(path):
                # Check for ephys data directory.
                if ephys_name in subdir:
                    ephys_subdir = subdir
                # Check for protocol directory.
                for protocol in protocol_names:
                    if protocol in subdir:
                        protocol_subdir = protocol
            # Need ephys and/or protocol data to do anything.
            if (ephys_subdir is None) & (protocol_subdir is None):
                raise Exception("Could not find ephys or protocol data.")
            # Parse SpikeGLX directory.
            nidaq_dir, imec_dir = self._parse_SGLX_dir(path + ephys_subdir)
            nidaq_metadata, nidaq_binary = read_NIDAQ(nidaq_dir)
            self.imec_spikes = []
            self.imec_sync = []
            for dir in imec_dir:
                imec_metadata, imec_binary, kilosort_array = read_Imec(dir)
                self.imec_spikes.append(ImecSpikes(imec_metadata, imec_binary, kilosort_array))
                self.imec_sync.append(ImecSync(imec_metadata, imec_binary, kilosort_array))
            # Create Population objects from spikes.
            self.populations = []
            for imec, sync in zip(self.imec_spikes, self.imec_sync):
                self.populations.append(Population(imec, sync))
            # Parse bTsS directory.
            btss_dir = path + protocol_subdir
            btss_visprot, btss_riglog = read_bTsS(btss_dir)
            self.btss_vstim = bTsSVStim(btss_riglog, btss_visprot)
            # Read sync channel from NI-DAQ.
            self.nidaq_sync = NIDAQSync(nidaq_metadata, nidaq_binary)
            # Read digital channels for stimuli, checking for blank with the bTsS protocol.
            for (key, val) in self.nidaq_digital.items():
                self.nidaq_digital[key] = NIDAQDigital(nidaq_metadata, 
                                                       nidaq_binary, 
                                                       digital_line_number=val["line_num"], 
                                                       blank=self.btss_vstim.vstim_params["blank_duration"])
            self.visual_stim = SparseNoise(sync=self.nidaq_sync, 
                                           pc_clock=self.nidaq_digital["pc_clock"], 
                                           photodiode=self.nidaq_digital["photodiode"], 
                                           btss_obj=self.btss_vstim)
            
            


    def _parse_SGLX_dir(
            self, 
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
        return nidaq_path, imec_path


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
    # Read Kilosort output files into numpy array.
    kilosort_names = ["spike_clusters.npy", "spike_times.npy"]
    kilosort_array = []
    for name in kilosort_names:
        kilosort_array.append(np.load(imec_path + "/" + name))
    kilosort_array = np.array(kilosort_array).squeeze().T
    return imec_metadata, imec_binary, kilosort_array

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

#     # Return a multiplicative factor for converting 16-bit file data
#     # to voltage. This does not take gain into account. The full
#     # conversion with gain is:
#     #         dataVolts = dataInt * fI2V / gain
#     # Note that each channel may have its own gain.
#     #
#     def Int2Volts(meta):
#         if meta['typeThis'] == 'imec':
#             if 'imMaxInt' in meta:
#                 maxInt = int(meta['imMaxInt'])
#             else:
#                 maxInt = 512
#             fI2V = float(meta['imAiRangeMax'])/maxInt
#         else:
#             fI2V = float(meta['niAiRangeMax'])/32768
#         return(fI2V)


#     # Return array of original channel IDs. As an example, suppose we want the
#     # imec gain for the ith channel stored in the binary data. A gain array
#     # can be obtained using ChanGainsIM(), but we need an original channel
#     # index to do the lookup. Because you can selectively save channels, the
#     # ith channel in the file isn't necessarily the ith acquired channel.
#     # Use this function to convert from ith stored to original index.
#     # Note that the SpikeGLX channels are 0 based.
#     #
#     def OriginalChans(meta):
#         if meta['snsSaveChanSubset'] == 'all':
#             # output = int32, 0 to nSavedChans - 1
#             chans = np.arange(0, int(meta['nSavedChans']))
#         else:
#             # parse the snsSaveChanSubset string
#             # split at commas
#             chStrList = meta['snsSaveChanSubset'].split(sep=',')
#             chans = np.arange(0, 0)  # creates an empty array of int32
#             for sL in chStrList:
#                 currList = sL.split(sep=':')
#                 if len(currList) > 1:
#                     # each set of contiguous channels specified by
#                     # chan1:chan2 inclusive
#                     newChans = np.arange(int(currList[0]), int(currList[1])+1)
#                 else:
#                     newChans = np.arange(int(currList[0]), int(currList[0])+1)
#                 chans = np.append(chans, newChans)
#         return(chans)





#     # Return gain for ith channel stored in nidq file.
#     # ichan is a saved channel index, rather than the original (acquired) index.
#     #
#     def ChanGainNI(ichan, savedMN, savedMA, meta):
#         if ichan < savedMN:
#             gain = float(meta['niMNGain'])
#         elif ichan < (savedMN + savedMA):
#             gain = float(meta['niMAGain'])
#         else:
#             gain = 1    # non multiplexed channels have no extra gain
#         return(gain)


#     # Return gain for imec channels.
#     # Index into these with the original (acquired) channel IDs.
#     #
#     def ChanGainsIM(meta):
#         imroList = meta['imroTbl'].split(sep=')')
#         # One entry for each channel plus header entry,
#         # plus a final empty entry following the last ')'
#         nChan = len(imroList) - 2
#         APgain = np.zeros(nChan)        # default type = float
#         LFgain = np.zeros(nChan)
#         if 'imDatPrb_type' in meta:
#             probeType = meta['imDatPrb_type']
#         else:
#             probeType = 0
#         if (probeType == 21) or (probeType == 24):
#             # NP 2.0; APGain = 80 for all AP
#             # return 0 for LFgain (no LF channels)
#             APgain = APgain + 80
#         else:
#             # 3A, 3B1, 3B2 (NP 1.0)
#             for i in range(0, nChan):
#                 currList = imroList[i+1].split(sep=' ')
#                 APgain[i] = currList[3]
#                 LFgain[i] = currList[4]
#         return(APgain, LFgain)


#     # Having accessed a block of raw nidq data using makeMemMapRaw, convert
#     # values to gain-corrected voltage. The conversion is only applied to the
#     # saved-channel indices in chanList. Remember, saved-channel indices are
#     # in the range [0:nSavedChans-1]. The dimensions of dataArray remain
#     # unchanged. ChanList examples:
#     # [0:MN-1]  all MN channels (MN from ChannelCountsNI)
#     # [2,6,20]  just these three channels (zero based, as they appear in SGLX).
#     #
#     def GainCorrectNI(self, dataArray, chanList, meta):
#         MN, MA, XA, DW = self.ChannelCountsNI(meta)
#         fI2V = Int2Volts(meta)
#         # make array of floats to return. dataArray contains only the channels
#         # in chanList, so output matches that shape
#         convArray = np.zeros(dataArray.shape, dtype=float)
#         for i in range(0, len(chanList)):
#             j = chanList[i]             # index into timepoint
#             conv = fI2V/ChanGainNI(j, MN, MA, meta)
#             # dataArray contains only the channels in chanList
#             convArray[i, :] = dataArray[i, :] * conv
#         return(convArray)


#     # Having accessed a block of raw imec data using makeMemMapRaw, convert
#     # values to gain corrected voltages. The conversion is only applied to
#     # the saved-channel indices in chanList. Remember saved-channel indices
#     # are in the range [0:nSavedChans-1]. The dimensions of the dataArray
#     # remain unchanged. ChanList examples:
#     # [0:AP-1]  all AP channels
#     # [2,6,20]  just these three channels (zero based)
#     # Remember that for an lf file, the saved channel indices (fetched by
#     # OriginalChans) will be in the range 384-767 for a standard 3A or 3B probe.
#     #
#     def GainCorrectIM(self, dataArray, chanList, meta):
#         # Look up gain with acquired channel ID
#         chans = self.OriginalChans(meta)
#         APgain, LFgain = self.ChanGainsIM(meta)
#         nAP = len(APgain)
#         nNu = nAP * 2

#         # Common conversion factor
#         fI2V = Int2Volts(meta)

#         # make array of floats to return. dataArray contains only the channels
#         # in chanList, so output matches that shape
#         convArray = np.zeros(dataArray.shape, dtype='float')
#         for i in range(0, len(chanList)):
#             j = chanList[i]     # index into timepoint
#             k = chans[j]        # acquisition index
#             if k < nAP:
#                 conv = fI2V / APgain[k]
#             elif k < nNu:
#                 conv = fI2V / LFgain[k - nAP]
#             else:
#                 conv = 1
#             # The dataArray contains only the channels in chanList
#             convArray[i, :] = dataArray[i, :]*conv
#         return(convArray)
