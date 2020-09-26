import os, sys
import numpy as np
from PIL import Image
import io
import struct
from datetime import datetime, timedelta,date
import time
from matplotlib.dates import date2num, num2date
import colour_demosaicing
import skvideo.io
import re
import copy
import pickle
import pdb
import cv2
import progressbar as pb

# Create interface sr for reading seq files.
#   sr = seqIo_reader( fName )
# Create interface sw for writing seq files.
#   sw = seqIo_Writer( fName, header )
# Crop sub-sequence from seq file.
#   seqIo_crop( fName, 'crop', tName, frames )
# Extract images from seq file to target directory or array.
#   Is = seqIo_toImgs( fName, tDir=[],skip=1,f0=0,f1=np.inf,ext='' )
# Create seq file from an array or directory of images or from an AVI file. DONE
#   seqIo_frImgs( fName, fName,header,aviName=[],Is=[],sDir=[],name='I',ndig=5,f0=0,f1=1e6 )
# Convert seq file by applying imgFun(I) to each frame I.
#   seqIo( fName, 'convert', tName, imgFun, varargin )
# Replace header of seq file with provided info.
#   seqIo( fName, 'newHeader', info )
# Create interface sr for reading dual seq files.
#   sr = seqIo( fNames, 'readerDual', [cache] )

FRAME_FORMAT_RAW_GRAY = 100 #RAW
FRAME_FORMAT_RAW_COLOR = 200 #RAW
FRAME_FORMAT_JPEG_GRAY = 102 #JPG
FRAME_FORMAT_JPEG_COLOR = 201 #JPG
FRAME_FORMAT_MONOB = 101 #BRGB8
FRAME_FORMAT_MONOB_JPEG = 103 #JBRGB
FRAME_FORMAT_PNG_GRAY = 0x001 #PNG
FRAME_FORMAT_PNG_COLOR = 0x002 #PNG

#matlab equivalent fread
def fread(fid, nelements, dtype):

    """Equivalent to Matlab fread function"""

    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    if data_array.size==1:data_array=data_array[0]
    return data_array

def fwrite(fid,a,dtype=np.str):
    # assuming 8but ASCII for string
    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype
    if isinstance(a,np.ndarray):
        data_array = a.astype(dt)
    else:
        data_array = np.array(a).astype(dt)
    data_array.tofile(fid)

def tsSync(video_path, srTop, srFront):
    #tsSync
    name = video_path.split('/')[-1]
    srDepth=seqIo_reader(video_path + '/' + name + '_DepGr_Raw.seq')

    #read timestamp of individual frames
    tsTop = srTop.getTs()
    tsFront = srFront.getTs()
    tsDepth = srDepth.getTs()

    #check the version of the videos
    videoDateStr = re.search('[0-9]+_[0-9]+-[0-9]+-[0-9]+',name).group(0)
    videoDateNum =date2num(datetime.strptime(videoDateStr ,'%Y%m%d_%H-%M-%S'))
    videoRefNum = date2num(datetime.strptime('20150401_00-00-00','%Y%m%d_%H-%M-%S'))
    seqV = 1 if videoDateNum < videoRefNum else 2

    ##correlate timestamps from one view to another
    mapTs = {}
    if seqV ==1:
        for f in range(len(tsDepth)):
            if tsDepth[f] - np.floor(tsDepth[f])>=.5: # Santiago's bug acquisistion software
                tsDepth[f]-= 1
        tsDepth-= 0.03 # substract the systematic timeshift

        #load front and top view and convert from UTC to PST
        hourShift = np.round((tsDepth[0]-tsTop[0])/3600.)*3600
        timeShift = tsDepth[0] - tsTop[0] - .066
        # print('tsDepth[0] - tsTop[0] = timeShift: %s sec'% str(timeShift - hourShift))
        tsTop +=  timeShift
        tsFront += timeShift

        srTop.ts = tsTop
        srFront.ts = tsFront
        srDepth.ts = tsDepth

        # Convert timestamps from left to right
        mapTs['T2F'] = transformTs(tsTop, tsFront)
        mapTs['F2T'] = transformTs(tsFront, tsTop)
        mapTs['T2D'] = transformTs(tsTop, tsDepth)
        mapTs['D2T'] = transformTs(tsDepth, tsTop)
        mapTs['F2D'] = transformTs(tsFront, tsDepth)
        mapTs['D2F'] = transformTs(tsDepth, tsFront)
    else:
        T = len(tsTop)
        F = len(tsFront)
        D = len(tsDepth)
        mapTs['T2F'] = resizeTs(T,F)
        mapTs['F2T'] = resizeTs(F,T)
        mapTs['T2D'] = resizeTs(T,D)
        mapTs['D2T'] = resizeTs(D,T)
        mapTs['F2D'] = resizeTs(F,D)
        mapTs['D2F'] = resizeTs(D,F)

    #display the time in string format
    # fTp=5
    # fFr = mapTs['T2F'][fTp]
    # fDp = mapTs['T2D'][fTp]
    # tF = ts2str(tsFront[fFr])
    # tT = ts2str(tsTop[fTp])
    # tD = ts2str(tsDepth[fDp])
    #
    # print('mapping to tTop:')
    # print('tTop   = ' + str(fTp) + ' - ' + tT)
    # print('tFront = ' + str(fFr) + ' - ' + tF)
    # print('tDepth = ' + str(fDp) + ' - ' + tD)

    # fDp = np.round(len(tsDepth)/1.5).astype(int)
    # fTp = mapTs['D2T'][fDp]
    # tT = ts2str(tsTop[fTp])
    # tD = ts2str(tsDepth[fDp])
    # print('mapping to tTop:')
    # print('tTop   = ' + str(fTp) + '. ' + tT)
    # print('tDepth = ' + str(fDp) + '. ' + tD)

    return mapTs,srTop,srFront

def transformTs(ts1, ts2):
    # map ts1 to ts2 where ts2 is the reference
    rankTs = np.zeros((len(ts1),4))
    for f in range(len(ts1)):
        tsDiff = ts2-ts1[f]
        tsRank = np.sort(abs(tsDiff))
        ind = np.argsort(abs(tsDiff))
        rankTs[f,:] = [ind[0]+1,ind[1]+1,tsRank[0],tsRank[1]]

    mapTs = np.round(smooth(rankTs[:,0],7)).astype(int)
    mapTs = np.round(smooth(mapTs,7)).astype(int)
    return mapTs

def smooth(a, WSZ):
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))

def resizeTs(t1, t2):
    if t1>t2:
        mapTs = np.hstack((np.array(range(t2))+1,np.ones((t1-t2),int)*t2))
    else:
        mapTs = np.array(range(t1))+1

    return mapTs

def ts2str(ts):
    t = ts / 86400. + date.toordinal(date(1971, 1, 2))
    # datetime.fromtimestamp(t)
    str_time = (datetime.fromordinal(int(t)) + timedelta(days=t % 1) - timedelta(days=366)).strftime(
        "%Y-%m-%d %H:%M:%S") + '.%03d' % np.round((ts - np.floor(ts)) * 1000)
    return str_time

def parse_ann(f_ann):
    header = 'Caltech Behavior Annotator - Annotation File'
    conf = 'Configuration file:'
    fid = open(f_ann)
    ann = fid.read().splitlines()
    fid.close()
    NFrames = []
    # check the header
    assert ann[0].rstrip() == header
    assert ann[1].rstrip() == ''
    assert ann[2].rstrip() == conf
    # parse action list
    l = 3
    names = [None] * 1000
    keys = [None] * 1000
    types = []
    bnds = []
    k = -1

    # get config keys and names
    while True:
        ann[l] = ann[l].rstrip()
        if not isinstance(ann[l], str) or not ann[l]:
            l += 1
            break
        values = ann[l].split()
        k += 1
        names[k] = values[0]
        keys[k] = values[1]
        l += 1
    names = names[:k + 1]
    keys = keys[:k + 1]

    # read in each stream in turn until end of file
    bnds0 = [None] * 10000
    types0 = [None] * 10000
    actions0 = [None] * 10000
    nStrm1 = 0
    while True:
        ann[l] = ann[l].rstrip()
        nStrm1 += 1
        t = ann[l].split(":")
        l += 1
        ann[l] = ann[l].rstrip()
        assert int(t[0][1]) == nStrm1
        assert ann[l] == '-----------------------------'
        l += 1
        bnds1 = np.ones((10000, 2), dtype=int)
        types1 = np.ones(10000, dtype=int) * -1
        actions1 = [None] * 10000
        k = 0
        # start the annotations
        while True:
            ann[l] = ann[l].rstrip()
            t = ann[l]
            if not isinstance(t, str) or not t:
                l += 1
                break
            t = ann[l].split()
            type = [i for i in range(len(names)) if t[2] == names[i]]
            type = type[0]
            if type == None:
                print('undefined behavior' + t[2])
            if bnds1[k - 1, 1] != int(t[0]) - 1 and k > 0:
                print('%d ~= %d' % (bnds1[k, 1], int(t[0]) - 1))
            bnds1[k, :] = [int(t[0]), int(t[1])]
            types1[k] = type
            actions1[k] = names[type]
            k += 1
            l += 1
            if l == len(ann):
                break
        if nStrm1 == 1:
            nFrames = bnds1[k - 1, 1]
        assert nFrames == bnds1[k - 1, 1]
        bnds0[nStrm1 - 1] = bnds1[:k]
        types0[nStrm1 - 1] = types1[:k]
        actions0[nStrm1 - 1] = actions1[:k]
        if l == len(ann):
            break
        while not ann[l]:
            l += 1

    bnds = bnds0[:nStrm1]
    types = types0[:nStrm1]
    actions = actions0[:nStrm1]

    idx = 0
    if len(actions[0]) < len(actions[1]):
        idx = 1
    type_frame = []
    action_frame = []
    len_bnd = []

    for i in range(len(bnds[idx])):
        numf = bnds[idx][i, 1] - bnds[idx][i, 0] + 1
        len_bnd.append(numf)
        action_frame.extend([actions[idx][i]] * numf)
        type_frame.extend([types[idx][i]] * numf)

    ann_dict = {
        'keys': keys,
        'behs': names,
        'nstrm': nStrm1,
        'nFrames': nFrames,
        'behs_se': bnds,
        'behs_dur': len_bnd,
        'behs_bout': actions,
        'behs_frame': action_frame
    }

    return ann_dict

def parse_ann_dual(f_ann):
    header = 'Caltech Behavior Annotator - Annotation File'
    conf = 'Configuration file:'
    fid = open(f_ann)
    ann = fid.read().splitlines()
    fid.close()
    NFrames = []
    # check the header
    assert ann[0].rstrip() == header
    assert ann[1].rstrip() == ''
    assert ann[2].rstrip() == conf
    # parse action list
    l = 3
    names = [None] * 1000
    keys = [None] * 1000
    types = []
    bnds = []
    k = -1

    # get config keys and names
    while True:
        ann[l] = ann[l].rstrip()
        if not isinstance(ann[l], str) or not ann[l]:
            l += 1
            break
        values = ann[l].split()
        k += 1
        names[k] = values[0]
        keys[k] = values[1]
        l += 1
    names = names[:k + 1]
    keys = keys[:k + 1]

    # read in each stream in turn until end of file
    bnds0 = [None] * 10000
    types0 = [None] * 10000
    actions0 = [None] * 10000
    nStrm1 = 0
    while True:
        ann[l] = ann[l].rstrip()
        nStrm1 += 1
        t = ann[l].split(":")
        l += 1
        ann[l] = ann[l].rstrip()
        assert int(t[0][1]) == nStrm1
        assert ann[l] == '-----------------------------'
        l += 1
        bnds1 = np.ones((10000, 2), dtype=int)
        types1 = np.ones(10000, dtype=int) * -1
        actions1 = [None] * 10000
        k = 0
        # start the annotations
        while True:
            ann[l] = ann[l].rstrip()
            t = ann[l]
            if not isinstance(t, str) or not t:
                l += 1
                break
            t = ann[l].split()
            type = [i for i in range(len(names)) if t[2] == names[i]]
            type = type[0]
            if type == None:
                print('undefined behavior' + t[2])
            if bnds1[k - 1, 1] != int(t[0]) - 1 and k > 0:
                print('%d ~= %d' % (bnds1[k, 1], int(t[0]) - 1))
            bnds1[k, :] = [int(t[0]), int(t[1])]
            types1[k] = type
            actions1[k] = names[type]
            k += 1
            l += 1
            if l == len(ann):
                break
        if nStrm1 == 1:
            nFrames = bnds1[k - 1, 1]
        assert nFrames == bnds1[k - 1, 1]
        bnds0[nStrm1 - 1] = bnds1[:k]
        types0[nStrm1 - 1] = types1[:k]
        actions0[nStrm1 - 1] = actions1[:k]
        if l == len(ann):
            break
        while not ann[l]:
            l += 1

    bnds = bnds0[:nStrm1]
    types = types0[:nStrm1]
    actions = actions0[:nStrm1]

    idx = 0
    if len(actions[0]) < len(actions[1]):
        idx = 1
    type_frame = []
    action_frame = []
    len_bnd = []


    for i in range(len(bnds[idx])):
        numf = bnds[idx][i, 1] - bnds[idx][i, 0] + 1
        len_bnd.append(numf)
        action_frame.extend([actions[idx][i]] * numf)
        type_frame.extend([types[idx][i]] * numf)


    type_frame2 = []
    action_frame2 = []
    len_bnd2 = []
    idx=1 if idx==0 else 0

    for i in range(len(bnds[idx])):
        numf = bnds[idx][i, 1] - bnds[idx][i, 0] + 1
        len_bnd2.append(numf)
        action_frame2.extend([actions[idx][i]] * numf)
        type_frame2.extend([types[idx][i]] * numf)

    ann_dict = {
        'keys': keys,
        'behs': names,
        'nstrm': nStrm1,
        'nFrames': nFrames,
        'behs_se': bnds,
        'behs_dur': len_bnd,
        'behs_bout': actions,
        'behs_frame': action_frame if 'interaction' not in action_frame else action_frame2,
        'behs_frame2': action_frame2 if 'interaction' in action_frame2 else action_frame
    }

    return ann_dict

def syncTopFront(f,num_frames,num_framesf):
    return int(round(f / (num_framesf - 1) * (num_frames - 1))) if num_framesf > num_frames else int(round(f / (num_frames - 1) * (num_framesf - 1)))



class seqIo_reader():
    def __init__(self,filename,info=[]):
        self.filename = filename
        try:
            self.file=open(filename,'rb')
        except EnvironmentError as e:
            print(os.strerror(e.errno))
        self.header={}
        self.seek_table=None
        self.frames_read=-1
        self.timestamp_length = 10
        if info==[]:
            self.readHeader()
        else:
            info.numFrames=0
        self.buildSeekTable(False)


    def readHeader(self):
        #make sure we do this at the beginning of the file
        assert self.frames_read == -1, "Can only read header from beginning of file"
        self.file.seek(0,0)
        # pdb.set_trace()

        # Read 1024 bytes (len of header)
        tmp = fread(self.file,1024,np.uint8)
        #check that the header is not all 0's
        n=len(tmp)
        if n<1024:raise ValueError('no header')
        if all(tmp==0): raise ValueError('fully empty header')
        self.file.seek(0,0)
        #first 4 bytes stor 0XFEED next 24 store 'Norpix seq '
        magic_number = fread(self.file,1,np.uint32)
        name = fread(self.file,10,np.uint16)
        name = ''.join(map(chr,name))
        if not '{0:X}'.format(magic_number)=='FEED' or not name=='Norpix seq':raise ValueError('invalid header')
        self.file.seek(4,1)
        #next 8 bytes for version and header size (1024) then 512 for desc
        version = int(fread(self.file,1,np.int32))
        hsize =int(fread(self.file,1,np.uint32))
        assert(hsize)==1024 ,"incorrect header size"
        # d = self.file.read(512)
        descr=fread(self.file,256,np.uint16)
        # descr = ''.join(map(chr,descr))
        # descr = ''.join(map(unichr,descr)).replace('\x00',' ')
        descr = ''.join([chr(x) for x in descr]).replace('\x00',' ')
        # descr = descr.encode('utf-8')
        #read more info
        tmp = fread(self.file,9,np.uint32)
        assert tmp[7]==0, "incorrect origin"
        fps = fread(self.file,1,np.float64)
        codec = 'imageFormat' + '%03d'%tmp[5]
        desc_format = fread(self.file,1,np.uint32)
        padding = fread(self.file,428,np.uint8)
        padding = ''.join(map(chr,padding))
        #store info
        self.header={'magicNumber':magic_number,
                     'name':name,
                     'seqVersion': version,
                     'headerSize':hsize,
                     'descr': descr,
                     'width':int(tmp[0]),
                     'height':int(tmp[1]),
                     'imageBitDepth':int(tmp[2]),
                     'imageBitDepthReal':int(tmp[3]),
                     'imageSizeBytes':int(tmp[4]),
                     'imageFormat':int(tmp[5]),
                     'numFrames':int(tmp[6]),
                     'origin':int(tmp[7]),
                     'trueImageSize':int(tmp[8]),
                     'fps':fps,
                     'codec':codec,
                     'descFormat':desc_format,
                     'padding':padding,
                     'nHiddenFinalFrames':0
                     }
        assert(self.header['imageBitDepthReal']==8)
        # seek to end fo header
        self.file.seek(432,1)
        self.frames_read += 1

        self.imageFormat = self.header['imageFormat']
        if self.imageFormat in (100,200):   self.ext = 'raw'
        elif self.imageFormat in (102,201): self.ext = 'jpg'
        elif self.imageFormat in(0x001,0x002):  self.ext = 'png'
        elif self.imageFormat == 101:       self.ext = 'brgb8'
        elif self.imageFormat == 103:       self.ext = 'jbrgb'
        else:                              raise ValueError('uknown format')

        self.compressed = True if self.ext in ['jpg','jbrgb','png','brgb8'] else False
        self.bit_depth = self.header['imageBitDepth']

        # My code uses a timestamp_length of 10 bytes, old uses 8. Check if not 10
        if self.bit_depth / 8 * (self.header['height'] * self.header['width']) + self.timestamp_length \
                != self.header['trueImageSize']:
            # If not 10, adjust to actual (likely 8) and print message
            self.timestamp_length = int(self.header['trueImageSize'] \
                                        - (self.bit_depth / 8 * (self.header['height'] * self.header['width'])))

    def buildSeekTable(self,memoize=False):
        """Build a seek table containing the offset and frame size for every frame in the video."""
        pickle_name = self.filename.strip(".seq") + ".seek"
        if memoize:
            if os.path.isfile(pickle_name):
                self.seek_table = pickle.load(open(pickle_name, 'rb'))
                return

        # assert self.header['numFrames']>0
        n=self.header['numFrames']
        if n==0:n=1e7

        seek_table = np.zeros((n)).astype(int)
        seek_table[0]=1024
        extra = 8 # extra bytes after image data , 8 for ts then 0 or 8 empty
        self.file.seek(1024,0)
        #compressed case

        if self.compressed:
            i=1
            while (True):
                try:
                    # size = fread(self.file,1,np.uint32)
                    # offset = seek_table[i-1] + size +extra
                    # seek_table[i]=offset
                    # # seek_table[i-1,1]=size
                    # self.file.seek(size-4+extra,1)

                    size = fread(self.file, 1, np.uint32)
                    offset = seek_table[i - 1] + size + extra
                    # self.file.seek(size-4+extra,1)
                    self.file.seek(offset, 0)
                    if i == 1:
                        if fread(self.file, 1, np.uint32) != 0:
                            self.file.seek(-4, 1)
                        else:
                            extra += 8;
                            offset += 8
                            self.file.seek(offset, 0)

                    seek_table[i] = offset
                    # seek_table[i-1,1]=size
                    i+=1
                except:
                    break
                    #most likely EOF
        else:
            #uncompressed case
            assert (self.header['numFrames']>0)
            frames = range(0, self.header["numFrames"])
            offsets = [x * self.header["trueImageSize"] + 1024 for x in frames]
            for i,offset in enumerate(offsets):
                seek_table[i]=offset
                # seek_table[i,1]=self.header["imageSize"]
        if n==1e7:
            n = np.minimum(n,i)
            self.seek_table=seek_table[:n]
            self.header['numFrames']=n
        else:
            self.seek_table=seek_table
        if memoize:
            pickle.dump(seek_table,open(pickle_name,'wb'))

        #compute frame rate from timestamps as stored fps may be incorrect
        # if n==1: return
        self.getTs()
        # ds = self.ts[1:100]-self.ts[:99]
        # ds = ds[abs(ds-np.median(ds))<.005]
        # if bool(np.prod(ds)): self.header['fps']=1/np.mean(ds)

    def getTs(self, n=None):
        if n==None: n=self.header['numFrames']
        if self.seek_table is None:
            self.buildSeekTable()

        ts = np.zeros((n))
        for i in range(n):
            if not self.compressed: #uncompressed
                self.file.seek(1024 + i*self.header['trueImageSize']+self.header['imageSizeBytes'],0)
            else: #compressed
                self.file.seek(self.seek_table[i],0)
                self.file.seek(fread(self.file,1,np.uint32)-4,1)
            # print(i)
            ts[i]=fread(self.file,1,np.uint32)+fread(self.file,1,np.uint16)/1000.


        self.ts=ts
        return self.ts

    def getFrame(self,index,decode=True):
        #get frame image (I) and timestamp (ts) at which frame was recorded
        nch = self.header['imageBitDepth']/8
        if self.ext in ['raw','brgb8']: #read in an uncompressed image( assume imageBitDepthReal==8)
            shape = (self.header['height'], self.header['width'])
            self.file.seek(1024 + index*self.header['trueImageSize'],0)
            I = fread(self.file,self.header['imageSizeBytes'],np.uint8)

            if decode:
                if nch==1:
                    I=np.reshape(I,shape)
                else:
                    I=np.reshape(I,(shape,nch))
                if nch==3:
                    t=I[:,:,2]; I[:,:,2]=I[:,:,0]; I[:,:,1]=t
                if self.ext=='brgb8':
                    I= colour_demosaicing.demosaicing_CFA_Bayer_bilinear(I,'BGGR')

        elif self.ext in ['jpg','jbrgb']:
            self.file.seek(self.seek_table[index],0)
            nBytes = fread(self.file,1,np.uint32)
            data = fread(self.file,nBytes-4,np.uint8)
            if decode:
                I = Image.open(io.BytesIO(data))
                if self.ext == 'jbrgb':
                    I=colour_demosaicing.demosaicing_CFA_Bayer_bilinear(I,'BGGR')
            else:
                I = data

        elif self.ext=='png':
            self.file.seek(self.seek_table[index],0)
            nBytes = fread(self.file,1,np.uint32)
            I= fread(self.file,nBytes-4,np.uint8)
            if decode:
                I= np.array(I).transpose(range(I.shape,-1,-1))
        else: assert(False)
        ts = fread(self.file,1,np.uint32)+fread(self.file,1,np.uint16)/1000.
        return np.array(I), ts

    # Close the file
    def close(self):
        self.file.close()

class seqIo_writer():
    def __init__(self,filename,old_header):
        self.file = open(filename,'wb')
        self.file.seek(0,0)
        self.header=old_header

        #create space for header
        fwrite(self.file,np.zeros(1024).astype(int),np.uint8)

        assert(set(['width','height','fps','codec']).issubset(self.header.keys()))

        codec = self.header['codec']
        if   codec in ['monoraw', 'imageFormat100']:        self.frmt = 100;self.nch = 1;self.ext = 'raw'
        elif codec in ['raw', 'imageFormat200']:            self.frmt = 200;self.nch = 3;self.ext = 'raw'
        elif codec in ['monojpg', 'imageFormat102']:        self.frmt = 102;self.nch = 1;self.ext = 'jpg'
        elif codec in ['jpg', 'imageFormat201']:            self.frmt = 201;self.nch = 3;self.ext = 'jpg'
        elif codec in ['monopng', 'imageFormat001']:        self.frmt = 0x001;self.nch = 1;self.ext = 'png'
        elif codec in ['png', 'imageFormat002']:            self.frmt = 0x002;self.nch = 3;self.ext = 'png'
        else:                                               raise ValueError('unknown format')

        self.header['imageFormat']=self.frmt
        self.header['imageBitDepth']=8*self.nch
        self.header['imageBitDepthReal']=8
        nBytes = self.header['width']*self.header['height']*self.nch
        self.header['imageSizeBytes']=nBytes
        self.header['numFrames']=0
        self.header['trueImageSize']=nBytes + 6 +512-np.mod(nBytes+6,512)

    # Close the file
    def close(self):
        self.writeHeader()
        self.file.close()

    def writeHeader(self):
        self.file.seek(0,0)
        # first write 4 bytes to store 0XFEED, next 24 store 'Nrpix seq  '
        fwrite(self.file,int('FEED',16),np.uint32)
        name = np.array(['Norpix seq  ']).view(np.uint8)
        fwrite(self.file,name, np.uint16)
        # next 8 bytes for version (3) and header size (1024) then 512 for descr
        fwrite(self.file,[3,1024],np.int32)
        if not 'descr' in self.header.keys() or len(np.array([self.header['descr']]).view(np.uint8))>256: d = np.array(['No Description']).view(np.uint8)
        else: d= np.array([self.header['descr']]).view(np.uint8)
        d = np.concatenate((d[:np.minimum(256,len(d))],np.zeros(256-len(d)).astype(np.uint8)))
        fwrite(self.file,d,np.uint16)
        #write remaining info
        vals= [self.header['width'],self.header['height'],self.header['imageBitDepth'],self.header['imageBitDepthReal'],
               self.header['imageSizeBytes'],self.header['imageFormat'],self.header['numFrames'],0,self.header['trueImageSize']]
        fwrite(self.file,vals,np.uint32)
        #store frame rate nad pad with 0s
        fwrite(self.file,self.header['fps'],np.float64)
        fwrite(self.file,np.zeros(432),np.uint8)

    def addFrame(self,I,ts=0,encode=1):
        nCh = self.header['imageBitDepth']/8
        ext = self.ext
        c = self.header['numFrames']+1
        if encode:
            siz = [self.header['height'],self.header['width'],nCh]
            assert(I.shape[0]==siz[0] and I.shape[1]==siz[1])
            if len(I.shape)==3:
                assert(I.shape[2]==siz[2] or I.shape[2]==self.nch)
        if ext=='raw':
            #write uncompressed image and assume imageBitDepthReal==8
            if not encode : assert(I.size==self.header['imageSizeBytes'])
            else:
                if nCh==3: t=I[:,:,2]; I[:,:,2]=I[:,:,0];I[:,:,0]=t
                if nCh==1: I=I.transpose()
                else: I = np.transpose( np.expand_dims(I, axis=2), (2, 1, 0) )
            # I= I.flat.view(np.uint8)
            I= I.flat
            fwrite(self.file,I,np.uint8)
            pad = self.header['trueImageSize']-self.header['imageSizeBytes']-6
        if ext =='jpg':
            if encode:
                #write red from to temporary jpg
                cv2.imwrite('tmp.jpg',I, [int(cv2.IMWRITE_JPEG_QUALITY ),80])
                # j=Image.fromarray(I.astype(np.uint8))
                # j.save('tmp.jpg')
                # I=Image.open('tmp.jpg')
                fid  = open('tmp.jpg','r')
                I = fid.read()
                fid.close()
                b=bytearray(I)
                assert (b[0] == 255 and b[1] == 216 and b[-2] == 255 and b[-1] == 217); # JPG
                os.remove('tmp.jpg')
                I = np.array(list(b)).astype(np.uint8)
            nbytes = len(I)+4
            fwrite(self.file,nbytes,np.uint32)
            # self.file.write(I)
            fwrite(self.file,I,np.uint8)
            pad = 10
        if ts==0: ts = (c-1)/self.header['fps']
        s = int(np.floor(ts))
        ms = int(np.round(np.mod(ts,1)*1000))
        fwrite(self.file,s,np.int32)
        fwrite(self.file,ms,np.uint16)
        self.header['numFrames']=c
        if pad>0:
            pad = np.zeros(pad).astype(np.uint8)
            fwrite(self.file,pad,np.uint8)

def seqIo_crop(fname, tname, frames):
    """
    Crop sub-sequence from seq file.
    
    Frame indices are 0 indexed. frames need not be consecutive and can
    contain duplicates. An index of -1 indicates a blank (all 0) frame. If
    contiguous subset of frames is cropped timestamps are preserved.
    
    USAGE
     seqIo( fName, 'crop', tName, frames )
    
    INPUTS
     fName      - seq file name
     tName      - cropped seq file name
     frames     - frame indices (0 indexed)
    """
    if not isinstance(frames, np.ndarray): frames=np.array(frames)
    sr = seqIo_reader(fname)
    sw = seqIo_writer(tname,sr.header)
    pad,_= sr.getFrame(0)
    pad = np.zeros(pad.size).astype(np.uint8)
    kp = frames>=0 & frames<sr.header['numFrames']
    if not np.all(kp): frames = frames[kp]
    print('%i out of bounds frames'% np.sum(~kp))
    ordered = np.all(frames[1:]==frames[:-1]+1)
    n= frames.size
    k=0
    for f in frames:
        if f<0:
            sw.addFrame(pad)
            continue
        I,ts = sr.getFrame(f)
        k+=1
        if ordered:
            sw.addFrame(I,ts)
        else:
            sw.addFrame(I)
    sr.close()
    sw.close

def seqIo_toImgs(fName, tDir=[], skip=1, f0=0, f1=np.inf, ext=''):
    """
    Extract images from seq file to target directory or array.
    
    USAGE
     Is = seqIo( fName, 'toImgs', [tDir], [skip], [f0], [f1], [ext] )
    
    INPUTS
     fName      - seq file name
     tDir       - [] target directory (if empty extract images to array)
     skip       - [1] skip between written frames
     f0         - [0] first frame to write
     f1         - [numFrames-1] last frame to write
     ext        - [] optionally save as given type (slow, reconverts)
    
    OUTPUTS
     Is         - if isempty(tDir) outputs image array (else Is=[])
    """
    sr = seqIo_reader(fName)
    f1 = np.minimum(f1,sr.header['numFrames']-1)
    frames = range(f0,f1,skip)
    n=len(frames)
    k=0
    #output images to array
    if tDir==[]:
        I,_=sr.getFrame(0)
        d = I.shape
        assert(len(d)==2 or len(d)==3)
        try:
            Is = np.zeros((I.shape+(n,))).astype(I.dtype)
        except:
            sr.close()
            raise
        for k in range(n):
            I,ts = sr.getFrame(k)
            if len(d)==2:
                Is[:,:,k]=I
            else:
                Is[:,:,:,k]=I
            print('saved %d' % k)

        sr.close()
    # output image directory
    if not os.path.exists(tDir):os.makedirs(tDir)
    if tDir.split('/')[-1]!='/':tDir+'/'
    Is = np.array([])
    for frame in frames:
        f = tDir + 'I%05.' % (frame)
        I, ts = sr.getFrame(frame)
        if ext!='':
            cv2.imwrite(f+ext,I)
        else:
            cv2.imwrite(f+sr.ext)
        k+=1
        print('saved %d' % frame)
    sr.close()
    return Is

def seqIo_frImgs(fName, header=[], aviName=[], Is=[], sDir=[], name='I', ndig=5, f0=0, f1=1e6):
    """
    Create seq file from an array or directory of images or from an AVI file.
    
    For info, if converting from array, only codec (e.g., 'jpg') and fps must
    be specified while width and height and determined automatically. If
    converting from AVI, fps is also determined automatically.
    
    USAGE
     seqIo( fName, 'frImgs', info, varargin )
    
    INPUTS
     fName      - seq file name
     info       - defines codec, etc, see seqIo>writer
     varargin   - additional params (struct or name/value pairs)
      .aviName    - [] if specified create seq from avi file
      .Is         - [] if specified create seq from image array
      .sDir       - [] source directory
      .skip       - [1] skip between frames
      .name       - ['I'] base name of images
      .nDigits    - [5] number of digits for filename index
      .f0         - [0] first frame to read
      .f1         - [10^6] last frame to read
    """
    
    if aviName!=[]: #avi movie exists
        vc = cv2.VideoCapture(aviName)
        if vc.isOpened(): rval = True
        else:
            rval = False
            print('video not readable')
            return
        fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
        NUM_FRAMES = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print(NUM_FRAMES)
        IM_TOP_H = vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        IM_TOP_W = vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        header['width']=IM_TOP_W
        header['height']=IM_TOP_H
        header['fps']=fps

        sw = seqIo_writer(fName,header)
        print('creating seq from AVI')
        # initialize timer
        timer = pb.ProgressBar(widgets=['Converting ', pb.Percentage(), ' -- ',
                                        pb.FormatLabel('Frame %(value)d'), '/',
                                        pb.FormatLabel('%(max)d'), ' [', pb.Timer(), '] ',
                                        pb.Bar(), ' (', pb.ETA(), ') '], maxval=NUM_FRAMES)
        for f in range(NUM_FRAMES):
            rval, im = vc.read()
            if rval:
                im= im.astype(np.uint8)
            sw.addFrame(im)
            timer.update(f)
        sw.close()
        timer.finish()
    elif Is==[]:
        assert(os.path.isdir(sDir))
        sw = seqIo_writer(fName,header)
        frmstr = '%s/%s%%0%ii.%s' % (sDir,name,ndig,header.ext)
        for frame in range(f0,f1):
            f = frmstr % frame
            if not os.path.isfile(f):break
            fid = open(f, 'r')
            if fid<0: sw.close();  assert(False)
            I = fid.read()
            fid.close()
            b = bytearray(I)
            assert (b[0] == 255 and b[1] == 216 and b[-2] == 255 and b[-1] == 217);  # JPG
            I = np.array(list(b)).astype(np.uint8)
            sw.addFrame(I,0,0)
        sw.close()
        if frame==f0: print('No images found')
    else:
        nd = len(Is.shape)
        if nd==2: nd=3
        assert(nd<=4)
        nFrm = Is.shape[nd-1]
        header['height']=Is.shape[0]
        header['width']=Is.shape[1]
        sw =seqIo_writer(fName,header)
        if nd==3:
            for f in range(nFrm): sw.addFrame(Is[:,:,f])
        if nd==4:
            for f in range(nFrm): sw.addFrame(Is[:,:,:,f])
        sw.close()

def seqIo_convert(fName, tName, imgFun, info=[], skip=1, f0=0, f1=np.inf):
    """
    Convert seq file by applying imgFun(I) to each frame I.
    
    USAGE
     seqIo( fName, 'convert', tName, imgFun, varargin )
    
    INPUTS
     fName      - seq file name
     tName      - converted seq file name
     imgFun     - function to apply to each image
     varargin   - additional params (struct or name/value pairs)
      .info       - [] info for target seq file
      .skip       - [1] skip between frames
      .f0         - [0] first frame to read
      .f1         - [inf] last frame to read
    """
    assert(fName!=tName)
    sr = seqIo_reader(fName)
    if info==[]: info=sr.header
    n=sr.header['numFrames']
    f1=np.minimum(f1,n-1)
    I,ts=sr.getFrame(0)
    I=imgFun(I)
    info['width']=I.shape[1]
    info['height']=I.shape[0]
    sw =seqIo_writer(tName,info)
    print('converting seq')
    for frame in range(f0,f1,skip):
        I, ts = sr.getFrame(frame)
        I = imgFun(I)
        if skip==1:
            sw.addFrame(I,ts)
        else:
            sw.addFrameI
    sw.close()
    sr.close()

def seqIo_newHeader(fName, info):
    """
    Replace header of seq file with provided info.
    
    Can be used if the file fName has a corrupt header. Automatically tries
    to compute number of frames in fName. No guarantees that it will work.
    
    USAGE
     seqIo( fName, 'newHeader', info )
    
    INPUTS
     fName      - seq file name
     info       - info for target seq file
    """
    d, n = os.path.split(fName)
    if d==[]:d='./'
    tName=fName[:-4] + '_new'  + time.strftime("%d_%m_%Y") + fName[-4:]
    sr = seqIo_reader(fName)
    sw = seqIo_writer(tName,info)
    n=sr.header['numFrames']
    for f in range(n):
        I,ts=sr.getFrame(f)
        sw.addFrame(I,ts)
    sr.close()
    sw.close()

class seqIo_dualReader():
    """
    seqIo_dualReader
    Create interface sr for reading dual seq files.
    
    Wrapper for two seq files of the same image dims and roughly the same
    frame counts that are treated as a single reader object. getframe()
    returns the concatentation of the two frames. For videos of different
    frame counts, the first video serves as the "dominant" video and the
    frame count of the second video is adjusted accordingly. Same general
    usage as in reader, but the only supported operations are: close(),
    getframe(), getinfo(), and seek().
    
    USAGE
     sr = seqIo( fNames, 'readerDual', [cache] )
    
    INPUTS
     fNames - two seq file names
     cache  - [0] size of cache (see seqIo>reader)
    
    OUTPUTS
     sr     - interface for reading seq file
    """
    def __init__(self,file1,file2):
        self.s1 = seqIo_reader(file1)
        self.s2 = seqIo_reader(file2)
        self.info  = self.s1.header
        #set the display to be vertically align
        self.info['height']=self.s1.header['height']+self.s2.header['height']
        self.info['width']=np.maximum(self.s1.header['width'],self.s2.header['width'])

        if self.s1.header['numFrames']!=self.s2.header['numFrames']:
            print('Two videos files have different number of frames')
            print('1st video has %d frames' % self.s1.header['numFrames'])
            print('2nd video has %d frames' % self.s2.header['numFrames'])
        print('first video %s is used as annotation refeence' % file1)

    def getFrame(self):
        I1,ts = self.s1.getFrame(0)
        I2,_ = self.s2.getFrame(0)

        w1 = I1.shape[1]
        w2 = I2.shape[1]

        if w1!=w2:
            m=np.argmax(w1,w2)
            if m==0:
                wl = int(np.floor((w1-w2)/2.))
                wr = w1-w2-wl
                nd = len(I2.shape)
                if nd==2:
                    padl = np.zeros((I2.shape[0],wl)).astype(np.uint8)
                    padr = np.zeros((I2.shape[0],wr)).astype(np.uint8)
                else:
                    padl = np.zeros((I2.shape[0],wl,I2.shape[2])).astype(np.uint8)
                    padr = np.zeros((I2.shape[0],wr,I2.shape[2])).astype(np.uint8)
                    I2 = np.concatenate((padl,I2,padr),axis=1)
            else:
                wl = int(np.floor((w2 - w1) / 2.))
                wr = w2 - w1 - wl
                nd = len(I2.shape)
                if nd == 2:
                    padl = np.zeros((I1.shape[0], wl)).astype(np.uint8)
                    padr = np.zeros((I1.shape[0], wr)).astype(np.uint8)
                else:
                    padl = np.zeros((I1.shape[0], wl, I1.shape[2])).astype(np.uint8)
                    padr = np.zeros((I1.shape[0], wr, I1.shape[2])).astype(np.uint8)
                    I1 = np.concatenate((padl, I1, padr), axis=1)
        I = np.hstack((I1,I2))
        return I,ts

class seqIo_extractor():
    """
    Create new seq files from top and fron view and syncronize them is not
    path_vid: video path
    vid_top: seq top video path and name
    vid_front: seq front video path and name
    s: start frame
    e: end frame

    """
    def __init__(self,path_vid,vid_top,vid_front,s,e):
        sr_top = seqIo_reader(path_vid+vid_top)
        sr_front = seqIo_reader(path_vid+vid_front)
        num_frames=sr_top.header['numFrames']
        num_framesf=sr_front.header['numFrames']
        name =os.path.dirname(video_top).split('/')[-1]

        if not os.path.exists(pathvid + name + '_%06d_%06d' % (s, e)):
            os.makedirs(pathvid + name + '_%06d_%06d' % (s, e))
        newdir = pathvid + name + '_%06d_%06d' % (s, e)
        video_out_top = newdir + '/' + name + '_%06d_%06d_Top_J85.seq' % (s, e)
        video_out_front = newdir + '/' + name + '_%06d_%06d_Front_J85.seq' % (s, e)

        sw_top = seqIo_writer(video_out_top, sr_top.header)
        sw_front = seqIo_writer(video_out_front, sr_front.header)

        for f in range(s - 1, e):
            if num_framesf > num_frames:
                I_top, ts = sr_top.getFrame(f2(f))
                I_front, ts2 = sr_front.getFrame(f)
            else:
                I_top, ts = sr_top.getFrame(f)
                I_front, ts2 = sr_front.getFrame(f2(f))
            sw_top.addFrame(I_top, ts)
            sw_front.addFrame(I_front, ts2)
            print(f)
        sw_top.close()
        sw_front.close()

    def f2(f):
            return int(round(f / (num_framesf - 1) * (num_frames - 1))) if num_framesf > num_frames else int(round(f / (num_frames - 1) * (num_framesf - 1)))

def seqIo_toVid(fName, ext='avi'):
    """
    seqIo_toVid
    Create seq file to another common used format as avi or mp4.
    
    USAGE
     seqIo( fName, ext )
    
    INPUTS
     fName      - seq file name
     ext        - video extension to convert to
    """

    assert fName[-3:]=='seq', 'Not a seq file'
    sr = seqIo_reader(fName)
    N  = sr.header['numFrames']
    h = sr.header['height']
    w = sr.header['width']
    fps = sr.header['fps']

    out = fName[:-3]+ext
    sw = skvideo.io.FFmpegWriter(out)
    # sw = cv2.VideoWriter(out, -1, fps, (w, h))
    timer = pb.ProgressBar(widgets=['Converting ', pb.Percentage(), ' -- ',
                                    pb.FormatLabel('Frame %(value)d'), '/',
                                    pb.FormatLabel('%(max)d'), ' [', pb.Timer(), '] ',
                                    pb.Bar(), ' (', pb.ETA(), ') '], maxval=N)

    for f in range(N):
        I, ts = sr.getFrame(f)
        sw.writeFrame(Image.fromarray(I))
        # sw.write(I)
        timer.update(f)
    timer.finish()
    # cv2.destroyAllWindows()
    # sw.release()
    sw.close()
    sr.close()
    print(out + ' converted')




# minimum header
# header = {'width': IM_TOP_W,
#           'height': IM_TOP_H,
#           'fps': fps,
#           'codec': 'imageFormat102'}
# filename= '/media/cristina/MARS_data/mice_project/teresa/Mouse156_20161017_17-22-09/Mouse156_20161017_17-22-09_Top_J85.seq'
# filename_out = filename[:-4] + '_new.seq'
# reader = seqIo_reader(filename)
# reader.header
# Initialize a SEQ writer
# writer = seqIo_writer(filename_out,reader.header)
# I,ts = reader.getFrame(0)
# writer.addFrame(I,ts)
# for f in range(8):
#     I,ts = reader.getFrame(f)
#     print(writer.file.tell())
#     writer.addFrame(I,ts)
# writer.close()
# reader.close()



