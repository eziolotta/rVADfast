from __future__ import division
import numpy 
import pickle
import os
import sys
import math
import code
from scipy.signal import lfilter
import speechproc
from copy import deepcopy
import scipy.io.wavfile as wav

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, 2019. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# 2017-12-02, Achintya Kumar Sarkar and Zheng-Hua Tan

# Usage: python rVAD_fast.py inWaveFile   
 

winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
ftThres=0.5; vadThres=0.4
opts=1

finwav=str(sys.argv[1])

#fvad=str(sys.argv[2])
output_root_dir = os.path.dirname(os.path.abspath(__file__))
fvad= os.path.join(output_root_dir, 'frame_vad.txt')


fs, data = speechproc.speech_wave(finwav)   
ft, flen, fsh10, nfr10, x_frames, signal =speechproc.sflux(data, fs, winlen, ovrlen, nftt)


# --spectral flatness --
pv01=numpy.zeros(nfr10)
pv01[numpy.less_equal(ft, ftThres)]=1 
pitch=deepcopy(ft)

pvblk=speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)


# --filtering--
ENERGYFLOOR = numpy.exp(-50)
b=numpy.array([0.9770,   -0.9770])
a=numpy.array([1.0000,   -0.9540])
fdata=lfilter(b, a, data, axis=0)


#--pass 1--
noise_samp, noise_seg, n_noise_samp=speechproc.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

#sets noisy segments to zero
for j in range(n_noise_samp):
    fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0 


vad_seg=speechproc.snre_vad(fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)

numpy.savetxt(fvad, vad_seg.astype(int),  fmt='%i')
print("%s --> %s " %(finwav, fvad))

###################################################
nb = 16
max_nb = float(2 ** (nb - 1))
###############
##TEST calcolo - SPLIT SEGNALE E SCRITTURA - funziona
_signal = signal * (max_nb + 1.0)
_signal = numpy.int16(_signal)
_signal = _signal[0:60000]
wav.write(os.path.join(output_root_dir,'test.wav' ),16000,_signal)
##############################
##################################
###SEGMENT AUDIO - WAV OUTPUT
speech_seg_data = []
curr_seg = None
for i in range(len(vad_seg)):
    is_speech = vad_seg[i]
    if(is_speech==0):
        ##is noise-silence segment
        if(curr_seg!=None):
            
            ##get flat array
            curr_seg = numpy.concatenate(curr_seg)
            ## calculate original signal
            curr_seg = curr_seg * (max_nb + 1.0)
            curr_seg = numpy.int16(curr_seg)
            speech_seg_data.append(curr_seg)
        curr_seg = None
    else:
        ## is speech segment
        curr_seg = [] if curr_seg==None else curr_seg
        c_data = x_frames[i] 
        ##append previous
        curr_seg.append(c_data)


#############################
##save wav
for i in range(len(speech_seg_data)):
    wav_data = speech_seg_data[i]
    output_file = os.path.join(output_root_dir, 'segment_{}.wav'.format(str(i)))
    wav.write(output_file,16000,wav_data)

######################


data=None; pv01=None; pitch=None; fdata=None; pvblk=None; vad_seg=None
     


