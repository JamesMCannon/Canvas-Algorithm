# import statements
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from canvas_alg_helper_funcs import get_vlfdata, resample, get_win, power_spectra, rebin_canvas, power_xspectra, time_avg  

# ----------------------------- Create a Test Signal -------------------------------
# create time domain data
fs = float(2**17)                      # sampling freq. 
sample_len = 5                         # seconds
t_vec = np.arange(0, sample_len, 1/fs) # create time vec
signal_freq = 4.98e3                   # signal freq.
amp = 2^8 - 18                         # signal amplitude

# channels (ex, ey, bx, by, bz)
shift = np.pi/2  # shift between 2 channels                       

bx = amp * np.sin(signal_freq * 2 * np.pi * t_vec)
by = amp * np.sin(signal_freq * 2 * np.pi * t_vec + shift)

# collect time domain channels here
channels_td = [bx, by]                

# plot a small chunk of time series data
plt.plot(t_vec[:100], bx[:100])
plt.plot(t_vec[:100], by[:100])
#plt.show()
plt.close()
# ------------------------------------------------------------------------------------

datadir = 'vlf_data/'
bx_raw, by_raw = get_vlfdata(datadir)
bx, by = resample([bx_raw, by_raw], 10, 100e3, fs)
channels_td = [bx, by]

# ---------------------------- Check 16bit Input Stimulus ----------------------------
# convert each channel of time series data to 16 bit (signed) integer
for ci, c in enumerate(channels_td):
    channels_td[ci] = np.int16(c)
# ------------------------------------------------------------------------------------


# ----------------------------------- Windowing --------------------------------------
# get hanning window coefs
nFFT = 1024
win = get_win(nFFT) # need to confirm bit output here
# ------------------------------------------------------------------------------------


# --------------------------- Break Up Time Domain Data -------------------------------
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
# Take FFT on input * window function for every 1024 points shifting by 512

# loop through all 512-length-segments -- this step accounts for the 50% overlap
channels_fd = [] # store channels now in frequency domain

for c in channels_td_segmented:
    c_fd = [] # store each segment of channel f domain

    # go through all 512-length segments
    for i in range(len(c)):
        if i == len(c)-1: # check if last segment
            break

        # grab current segment and next one -- THIS STEP COMBINES TWO 512-length into 1024 
        cs_2 = np.append(c[i], c[i+1]) 

        # mutitply elementwise by windowing func
        cs_win = np.multiply(win, cs_2) 

        # take FFT
        cs_f = np.fft.fft(cs_win)
        
        # save it
        c_fd.append(cs_f)

    # save the output for each channel - vector of 1024-pt FFTs
    channels_fd.append(c_fd) 

    # check 32 bit out here
# ------------------------------------------------------------------------------------


# ------------------------------ Power Calculation ------------------------------------
spectra = []
xspectra = []

# loop through the channels
xspec_done = []
for ci, c in enumerate(channels_fd): 
    # loop through the channels again
    for ci_x, c_x in enumerate(channels_fd): 
        # check if same channel or a diff one
        if ci == ci_x: 
            spectra.append(power_spectra(c))

        # different channel = x-spectra, but only need upper triangle
        if ci != ci_x and (ci, ci_x) not in xspec_done and (ci_x, ci) not in xspec_done: 
            Preal, Pimag = power_xspectra(c, c_x)
            xspectra.append(Preal)
            xspectra.append(Pimag)

# check output here! 64 bit
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

# check output here -- still 64 bit
# ------------------------------------------------------------------------------------


# -------------------------------- Compression ---------------------------------------
# first, sqrt
spectra_sqrt = [np.sqrt(s) for s in spectra_favg]
xspectra_sqrt = [np.sqrt(xs) for xs in xspectra_favg]

# log 2 compression
spectra_compressed = [np.log2(sc) * 8 for sc in spectra_sqrt]
xspectra_compressed = [np.log2(xsc) * 8 for xsc in xspectra_sqrt]
# ------------------------------------------------------------------------------------


# -------------------------------- Spectrogram ---------------------------------------
t_size = np.shape(spectra_favg)[1]    # second dimension is number of time points
tt = np.arange(1,t_size+1,1)          # create a time vector
ff = fbins[::2]

fig, axs = plt.subplots(1, len(spectra))
plt.subplots_adjust(wspace=0.3,hspace=0.5)
for i, s in enumerate(spectra_sqrt):
    spectrogram_spec = np.array(s).T
    axs[i].pcolormesh(tt, ff, np.log10(spectrogram_spec), cmap = plt.cm.jet)
    axs[i].set_title('channel'+ str(i))

# set labels
axs[0].set_ylabel('freq [Hz]')
axs[0].set_xlabel('time [s]')
axs[1].set_xlabel('time [s]')
plt.show()
# ------------------------------------------------------------------------------------