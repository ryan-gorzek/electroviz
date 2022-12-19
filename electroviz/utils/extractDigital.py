
import numpy as np

# Return an array [lines X timepoints] of uint8 values for a
# specified set of digital lines.
#
# - dwReq is the zero-based index into the saved file of the
#    16-bit word that contains the digital lines of interest.
# - dLineList is a zero-based list of one or more lines/bits
#    to scan from word dwReq.
#
def extractDigital(rawData, firstSamp, lastSamp, dwReq, dLineList, meta):
    """"""

    # Return counts of each nidq or imec channel type that composes the timepoints
    # stored in the binary file.
    def getChannelCounts(meta):
        if meta['typeThis'] == 'imec':
            chanCountList = meta['snsApLfSy'].split(sep=',')
            AP = int(chanCountList[0])
            LF = int(chanCountList[1])
            SY = int(chanCountList[2])
            channelCounts = (AP, LF, SY)
        else:
            chanCountList = meta['snsMnMaXaDw'].split(sep=',')
            MN = int(chanCountList[0])
            MA = int(chanCountList[1])
            XA = int(chanCountList[2])
            DW = int(chanCountList[3])
            channelCounts = (MN, MA, XA, DW)
        return channelCounts

    # Get channel index of requested digital word dwReq
    if meta['typeThis'] == 'imec':
        AP, LF, SY = getChannelCounts(meta)
        if SY == 0:
            print("No imec sync channel saved.")
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = AP + LF + dwReq
    else:
        MN, MA, XA, DW = getChannelCounts(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = MN + MA + XA + dwReq

    selectData = np.ascontiguousarray(rawData[digCh, firstSamp:lastSamp+1], 'int16')
    nSamp = lastSamp-firstSamp + 1

    # unpack bits of selectData; unpack bits works with uint8
    # original data is int16
    bitWiseData = np.unpackbits(selectData.view(dtype='uint8'))
    # output is 1-D array, nSamp*16. Reshape and transpose
    bitWiseData = np.transpose(np.reshape(bitWiseData, (nSamp, 16)))

    nLine = len(dLineList)
    digArray = np.zeros((nLine, nSamp), 'uint8')
    for i in range(0, nLine):
        byteN, bitN = np.divmod(dLineList[i], 8)
        targI = byteN*8 + (7 - bitN)
        digArray[i, :] = bitWiseData[targI, :]
    return(digArray)
