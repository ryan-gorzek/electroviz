
# MIT License
# Copyright (c) 2022-3 Ryan Gorzek
# https://github.com/gorzek-ryan/electroviz/blob/main/LICENSE
# https://opensource.org/licenses/MIT

import numpy as np

def gainCorrectIM(dataArray, chanList, meta, path):


    def OriginalChans(meta):
        if meta['snsSaveChanSubset'] == 'all':
            # output = int32, 0 to nSavedChans - 1
            chans = np.arange(0, int(meta['nSavedChans']))
        else:
            # parse the snsSaveChanSubset string
            # split at commas
            chStrList = meta['snsSaveChanSubset'].split(sep=',')
            chans = np.arange(0, 0)  # creates an empty array of int32
            for sL in chStrList:
                currList = sL.split(sep=':')
                if len(currList) > 1:
                    # each set of contiguous channels specified by
                    # chan1:chan2 inclusive
                    newChans = np.arange(int(currList[0]), int(currList[1])+1)
                else:
                    newChans = np.arange(int(currList[0]), int(currList[0])+1)
                chans = np.append(chans, newChans)
        return(chans)


    def ChanGainsIM(meta):
        imroList = meta['imroTbl'].split(sep=')')
        # One entry for each channel plus header entry,
        # plus a final empty entry following the last ')'
        nChan = len(imroList) - 2
        APgain = np.zeros(nChan)        # default type = float
        LFgain = np.zeros(nChan)
        if 'imDatPrb_type' in meta:
            probeType = meta['imDatPrb_type']
        else:
            probeType = 0
        if (probeType == 21) or (probeType == 24):
            # NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            APgain = APgain + 80
        else:
            # 3A, 3B1, 3B2 (NP 1.0)
            for i in range(0, nChan):
                currList = imroList[i+1].split(sep=' ')
                APgain[i] = currList[3]
                LFgain[i] = currList[4]
        return(APgain, LFgain)


    def Int2Volts(meta):
        if meta['typeThis'] == 'imec':
            if 'imMaxInt' in meta:
                maxInt = int(meta['imMaxInt'])
            else:
                maxInt = 512
            fI2V = float(meta['imAiRangeMax'])/maxInt
        else:
            fI2V = float(meta['niAiRangeMax'])/32768
        return(fI2V)


    # Look up gain with acquired channel ID
    chans = OriginalChans(meta)
    APgain, LFgain = ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2

    # Common conversion factor
    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.memmap(path, shape=dataArray.shape, dtype='float', mode="w+")
    for i in range(0, len(chanList)):
        j = chanList[i]     # index into timepoint
        k = chans[j]        # acquisition index
        if k < nAP:
            conv = fI2V / APgain[k]
        elif k < nNu:
            conv = fI2V / LFgain[k - nAP]
        else:
            conv = 1
        # The dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :]*conv
    return(convArray)

