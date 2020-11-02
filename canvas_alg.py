# import statements
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from canvas_alg_helper_funcs import get_vlfdata, resample, get_win, power_spectra, rebin_canvas, power_xspectra, time_avg  

# ----------------------------- Create a Test Signal -------------------------------
# create time domain data
fs = float(2**17)                      # sampling freq. 
sample_len = 4                         # seconds
t_vec = np.arange(0, sample_len, 1/fs) # create time vec
signal_freq = 1e3                      # signal freq.
amp = 2^8 -18                          # signal amplitude
# is the window output correct for the low freq?
# channels (ex, ey, bx, by, bz)
shift = np.pi/2  # shift between 2 channels                       

bx = amp * np.sin(signal_freq * 2 * np.pi * t_vec)
by = amp * np.sin(signal_freq * 2 * np.pi * t_vec + shift)

# collect time domain channels here
channels_td = [bx, by]

# -----------------------------check input signal------------------------------------
plt_chk = 1024
"""
plt.plot(t_vec[:plt_chk], bx[:plt_chk])
plt.plot(t_vec[:plt_chk], by[:plt_chk])
plt.title('Input Signal')
plt.show()
plt.close()
"""
# ------------------------------------------------------------------------------------


# ----------------------or get input signal from VLF data-----------------------------
"""
datadir = 'vlf_data/'
bx_raw, by_raw = get_vlfdata(datadir)
sample_len = 2 # length of sample desired
sample_fs = 100e3 # frequency of sample
bx, by = resample([bx_raw, by_raw], sample_len, sample_fs, fs)
channels_td = [bx, by]

# -----------------------------check input signal------------------------------------
# plot a small chunk of time series data
plt_chk = 1024
plt.plot(t_vec[:plt_chk], bx[:plt_chk])
plt.plot(t_vec[:plt_chk], by[:plt_chk])
plt.title('Input Signal')
plt.show()
plt.close()
"""
# ------------------------------------------------------------------------------------


# ------------------------ Convert to 16bit Input Stimulus ---------------------------
# WHAT? make 32 input maybe?
# convert each channel of time series data to 16 bit (signed) integer
#for ci, c in enumerate(channels_td):
#    channels_td[ci] = np.int16(c)
# ------------------------------------------------------------------------------------


# ----------------------------------- Windowing --------------------------------------
# get hanning window coefs
nFFT = 1024
win = get_win(nFFT) # need to confirm bit output here

# -----------------------------check input window------------------------------------
"""
plt.plot(np.arange(0, len(win)), win)
plt.title('Input Window')
plt.show()
plt.close()
"""
# ------------------------------------------------------------------------------------


# ----------------------------Break Up Time Domain Data-------------------------------
# segmented time domain data
channels_td_segmented = []

# go through each channel
for c in channels_td: 
    
    # store each segmented channel
    c_segs = []  

    # loop through channel td data by 512
    for i in range(0, len(c), int(nFFT/2)): 

        # grab a segment
        cs = c[i:i+int(nFFT/2)] 

        # add 0s to segment that isn't 512 points
        while len(cs) < int(nFFT/2): 
            cs = np.append(cs, 0)

        # append each segment to segmented channel list
        c_segs.append(cs) 

    # all channels now segmented and stored here
    channels_td_segmented.append(c_segs)
# ------------------------------------------------------------------------------------

# ---------------------------- Perform FFT -------------------------------------------
# FFT on input * window for every 1024 points shifting by 512 - 50% overlap 
channels_fd = [] # store channels now in frequency domain

for c in channels_td_segmented:
    c_fd = [] # store each segment of channel f domain

    # go through all 512-length segments
    for i in range(len(c)):
        if i == len(c)-1: # check if last segment
            clast = np.zeros(np.size(c[i])) # add zeros for right after last segment
            # grab current segment and next one -- THIS STEP COMBINES TWO 512-length into 1024 
            cs_2 = np.append(c[i], clast) 
        else:
            cs_2 = np.append(c[i], c[i+1])

        # mutitply elementwise by windowing func
        cs_2 = np.array(cs_2)
        cs_win = np.multiply(win, cs_2) 

        # ---------------------------check win * input---------------------------------
        """
        if i==1: 
            plt.plot(np.arange(0, len(cs_win)), cs_win)
            plt.title('Input Window x Input Signal')
            #plt.show()
            plt.close()
        """
        # ----------------------------------------------------------------------------

        # take FFT
        cs_f = np.fft.fft(cs_win)

        # ---------------------------check FFT---------------------------------------
        center_freqs = np.fft.fftfreq(nFFT,d=1/fs)
        """
        if i==1:
            center_freqs = np.fft.fftfreq(nFFT,d=1/fs)
            plt.plot(center_freqs, cs_f)
            plt.title('FFT Spectrum')
            #plt.show()
            plt.close()
        """
        # ---------------------------------------------------------------------------
        
        # save it
        c_fd.append(cs_f)

    # save the output for each channel - vector of 1024-pt FFTs
    channels_fd.append(c_fd) 

# ------------------------------------------------------------------------------------


# ------------------------------ Power Calculation -----------------------------------
# QUESTION: c^2 in FPGA?
spectra = []
xspectra = []

# loop through the channels
for c in channels_fd: 
    spectra.append(power_spectra(c))

    # --------------------------check power calc-------------------------------------
    t_slice = 1
    plt.loglog(center_freqs, power_spectra(c)[t_slice])
    plt.title('Power Spectra')
    plt.show()
    plt.close()
    # -------------------------------------------------------------------------------


# loop through the channels and perform xspectra calcs
for i in range(0,len(channels_fd)):
    for j in range(i+1,len(channels_fd)):
        Preal, Pimag = power_xspectra(channels_fd[i], channels_fd[j])
        xspectra.append(Preal)
        xspectra.append(Pimag)
        
# ------------------------------------------------------------------------------------  


# -------------------------------------- Time Avg ------------------------------------
# time average for each spectra and cross spectra
spectra_tavg = [time_avg(s, nFFT, fs) for s in spectra]
xspectra_tavg = [time_avg(xs, nFFT, fs) for xs in xspectra]
# ------------------------------------------------------------------------------------ 


# ---------------------------- Rebin into CANVAS bins ---------------------------------
# parse text file with canvas bins
fname = 'fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]

# monotonic and 1D list of canvas fbins
fbins = [item for sublist in fbins_dbl for item in sublist]

# find center freqs of bins from fft
center_freqs = np.fft.fftfreq(nFFT,d=1/fs) # seems like missing last bin, but doesn't matter bc we discard it

# rebin to get average for canvas fbins from the time averaged spectra and xspecrta
spectra_favg = [rebin_canvas(s, fbins, center_freqs) for s in spectra_tavg]
xspectra_favg = [rebin_canvas(xs, fbins, center_freqs) for xs in xspectra_tavg]

# ------------------------------------------------------------------------------------


# -------------------------------- Compression ---------------------------------------
# sqrt the spectra
spectra_sqrt = [np.sqrt(s) for s in spectra_favg]

# log 2 compression -- 64 bit unsigned integer, can use the floor function
# talk about increasing number of bits to spectra --- think about in context of SVD
spectra_compressed = [np.log2(sc) * 8 for sc in spectra_sqrt]

# need to extract the sign, save it, compress, and put back the sign after decompression
xspectra_compressed = [np.log2(xsc) * 32 for xsc in xspectra_favg]
# ------------------------------------------------------------------------------------

# -------------------------------- Decompression ---------------------------------------
spectra_dc = [2**(sc / 8) for sc in spectra_compressed]
# include the sign back here
xspectra_dc = [2**(xsc / 32) for xsc in xspectra_compressed]
# ------------------------------------------------------------------------------------


# -------------------------------- Spectrogram ---------------------------------------
# doesnt seem to be working for only one second ....
t_size = np.shape(spectra_favg)[1]    # second dimension is number of time points
tt = np.arange(1,t_size+1,1)          # create a time vector

# parse text file with center canvas bins
fname = 'fbins_center.txt'                                 
fbins_c_str = np.genfromtxt(fname, dtype='str') 
fbins_center = [(float(f.replace(',',''))) for f in fbins_c_str]
# --------------------------check decomp calc------------------------------------
plt.loglog(fbins_center, spectra_dc[0][t_slice])
plt.title('Power Spectra After Decompression')
plt.show()
plt.close()
# -------------------------------------------------------------------------------

fig, axs = plt.subplots(1, len(spectra))
plt.subplots_adjust(wspace=0.3,hspace=0.5)

for i, s in enumerate(spectra_dc):
    s = np.array(s).T
    axs[i].pcolormesh(tt, np.log10(fbins_center), s, cmap = plt.cm.jet)
    axs[i].set_title('channel'+ str(i))

# set labels
axs[0].set_ylabel('freq [Hz]')
axs[0].set_xlabel('time [s]')
axs[1].set_xlabel('time [s]')
plt.show()
plt.close()
# ------------------------------------------------------------------------------------