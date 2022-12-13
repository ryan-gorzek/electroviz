# MIT License
# Copyright (c) 2022 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import os
import glob
import numpy as np

class NIDAQ
    """
    Interface for National Instruments DAQ signals acquired via SpikeGLX.
    These data are stored in binary files (**.nidq.bin) and have accompanying metadata files (**.niqd.meta).
    """

    def __init__(
            self,
            SpikeGLX_path,
            bTsS_path,
            buffer_len = 0,
            digital_lines = [7, 5, 4, 1],
            digital_names = ["sync", "camera", "pc_clock", "photodiode"],
            analog_lines = [],
            analog_names = [],
        ):
        """
        Read NIDAQ binary and metadata files (both must be present) from SpikeGLX and align them to bTsS stimuli.

        **.nidq.bin is read into a numpy memorymap.
        **.nidq.meta is read into a Python dictionary.

        SpikeGLX files are read by functions modified from SpikeGLX Datafile Tools created by Bill Karsh.
        bTsS files are read by function modified from bTsS created by Joao Couto.
        """

        # Load SpikeGLX binary and metadata files.
        assert os.path.exists(SpikeGLX_path), "Could not find the specified path to SpikeGLX data."
        # Locate and read the meta file into a Python dictionary.
        meta_file = glob.glob(SpikeGLX_path + "*.nidq.meta")
        assert len(meta_file) == 1, "The **.nidq.meta file could not be read properly, check that it exists in the path without conflicts.")
        self.meta_dict = self._SGLX_readMeta(meta_file)
        # Locate and read the binary file into a numpy memory map.
        bin_file = glob.glob(SpikeGLX_path + "*.nidq.bin")
        assert len(bin_file) == 1, "The **.nidq.bin file could not be read properly, check that it exists in the path without conflicts.")
        self.bin_memmap = self._SGLX_makeMemMapRaw(bin_file, self.meta_dict)

        # Load bTsS vislog (stimulus identity) and riglog (timestamp) files.
        assert os.path.exists(bTsS_path), "Could not find the specified path to bTsS data."
        # Locate and read the vislog file
        vislog_file = glob.glob(bTsS_path + "" + "*.vislog")
        assert len(vislog_file) == 1, "The **.vislog file could not be read properly, check that it exists in the path without conflicts.")

        # Locate and read the riglog file
        riglog_file = glob.glob(bTsS_path + "" + "*.riglog")
        assert len(riglog_file) == 1, "The **.riglog file could not be read properly, check that it exists in the path without conflicts.")

        #### Align -- take diff of pc_clock and photodiode (buffer matters here) and match onset/offset with bTsS timestamps

    # def get_stim_df()

    # get_channels

    # plot_channels

    def _align_stim(
            self, 
        ):
        """Align bTsS stimulus identities to PC clock and photodiode times."""

    def _parse_vislog(
            self,
            vislog_file, 
        ):
        """"""

    def _parse_riglog(
            self,
            riglog_file, 
        ):
        """"""

    def _SGLX_readMeta(
            metaFile, 
        ):
        """
        Parse ini file returning a dictionary whose keys are the metadata
        left-hand-side-tags, and values are string versions of the right-hand-side
        metadata values. We remove any leading '~' characters in the tags to match
        the MATLAB version of readMeta.

        The string values are converted to numbers using the "int" and "float"
        functions. Note that python 3 has no size limit for integers.
        """
        metaDict = {}
        with metaFile.open() as f:
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

    def _SGLX_makeMemMapRaw(
            binFile, 
            metaDict, 
        ):
        """
        Read **.nidq.bin as a numpy memory map (channels x time).
        """
        nChan = int(metaDict['nSavedChans'])
        nFileSamp = int(int(metaDict['fileSizeBytes'])/(2*nChan))
        binMemMap = np.memmap(binFile, dtype='int16', mode='r',
                            shape=(nChan, nFileSamp), offset=0, order='F')
        return(binMemMap)

    def _SGLX_ChannelCountsNI(
            metaDict, 
        ):
        """
        Return counts of each nidq channel type that composes the timepoints
        stored in the binary file.
        """
        chanCountList = meta['snsMnMaXaDw'].split(sep=',')
        MN = int(chanCountList[0])
        MA = int(chanCountList[1])
        XA = int(chanCountList[2])
        DW = int(chanCountList[3])
        return(MN, MA, XA, DW)

    def _SGLX_ExtractDigital(
            binFile,
            firstSamp,
            lastSamp,
            dwReq,
            dLineList,
            metaDict,
        ):
        """
        Return an array [lines X timepoints] of uint8 values for a
        specified set of digital lines.
          - dwReq is the zero-based index into the saved file of the
            16-bit word that contains the digital lines of interest.
          - dLineList is a zero-based list of one or more lines/bits
            to scan from word dwReq.
        """
        # Get channel index of requested digital word dwReq.
        MN, MA, XA, DW = ChannelCountsNI(metaDict)
        if dwReq > DW-1:
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = MN + MA + XA + dwReq

        selectData = np.ascontiguousarray(binFile[digCh, firstSamp:lastSamp+1], 'int16')
        nSamp = lastSamp-firstSamp + 1
        # Unpack bits of selectData; unpack bits works with uint8.
        # Original data is int16.
        bitWiseData = np.unpackbits(selectData.view(dtype='uint8'))
        # Output is 1-D array, nSamp*16. Reshape and transpose.
        bitWiseData = np.transpose(np.reshape(bitWiseData, (nSamp, 16)))

        nLine = len(dLineList)
        digArray = np.zeros((nLine, nSamp), 'uint8')
        for i in range(0, nLine):
            byteN, bitN = np.divmod(dLineList[i], 8)
            targI = byteN*8 + (7 - bitN)
            digArray[i, :] = bitWiseData[targI, :]
        return(digArray)
