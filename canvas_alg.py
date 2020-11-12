# import statements
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from scipy.fftpack import fft, fftfreq, fftshift
from canvas_alg_helper_funcs import get_vlfdata, resample, get_win, power_spectra, rebin_canvas, power_xspectra, time_avg  

# ----------------------------- Create a Test Signal ------------------------------- 
# create time domain data
fs = 131072.                           # sampling freq. 
sample_len = 3.0                         # seconds
t_vec = np.linspace(0, sample_len, num=fs*sample_len)   # create time vec
signal_freq1 = 8.3e3                     # signal freq. 1
signal_freq2 = 598                     # signal freq. 2
amp = 249.                             # signal amplitude -- SIGNED 

# channels (ex, ey, bx, by, bz)
# shift = np.pi/2  # shift between 2 channels                       

bx = amp * np.sin(signal_freq1 * 2 * np.pi * t_vec)
by = amp * np.sin(signal_freq2 * 2 * np.pi * t_vec)
bx_check = bx
by_check = by
check = [bx_check, by_check]

# make an integer
bx = [round(bxx,0) for bxx in bx]
by = [round(byy,0) for byy in by]

# collect time domain channels here
channels_td = [bx, by]

plt_chk = int(1e3)
for ch, ch_check in zip(channels_td, check):
    plt.plot(t_vec[:plt_chk], ch[:plt_chk])
    plt.plot(t_vec[:plt_chk], ch_check[:plt_chk])
    plt.title('Input Signal - first 1024')
    plt.show()
    plt.close()

do_plots = True

# -----------------------------check input signal------------------------------------
plt_chk = 1024

if do_plots:
    for ch in channels_td:
        plt.plot(t_vec[:plt_chk], ch[:plt_chk])
    plt.title('Input Signal - first 1024')
    plt.show()
    plt.close()

# cast input (ints) to 16bit int represented in hex
with open('channel1_input.txt', 'w') as output:
    for b in bx[:1024]:
        output.write(format(np.int16(b) & 0xffff, '04X') + '\n')

with open('channel2_input.txt', 'w') as output:
    for b in by[:1024]:
        output.write(format(np.int16(b) & 0xffff, '04X') + '\n')
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

# ----------------------------------- Windowing --------------------------------------
# get hanning window coefs
nFFT = 1024
win = get_win(nFFT) # need to confirm bit output here

# -----------------------------check input window------------------------------------
if do_plots:
    plt.plot(np.arange(0, len(win)), win)
    plt.title('Input Window')
    plt.show()
    plt.close()
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
    
    # -------------------------------check -----------------------------------------------  
    if do_plots:
        plt.plot(np.arange(1,512+1),c_segs[0])
        plt.plot(np.arange(512+1,1024+1),c_segs[1])
        plt.title('Input Series - segmented (first 2 chunks)')
        plt.show()
        plt.close()

    # all channels now segmented and stored here
    channels_td_segmented.append(c_segs)

# ------------------------------------------------------------------------------------

# ---------------------------- Perform FFT -------------------------------------------
# FFT on input * window for every 1024 points shifting by 512 -- 50% overlap 
channels_fd = [] # store channels now in frequency domain

for ci, c in enumerate(channels_td_segmented):
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
        cs_win = np.multiply(win, cs_2) # should be integer (with max 2^31-1)

        # ---------------------------check win * input---------------------------------
        
        if i==1 and do_plots:
            plt.plot(np.arange(0, len(cs_win)), cs_win)
            plt.title('Input Window x Input Signal - First 1024')
            plt.show()
            plt.close()
        
        # ----------------------------------------------------------------------------

        # take FFT
        cs_f = fft(cs_win)

        # convert real and imag to int
        cs_f_r = [int(np.real(c_r)) for c_r in cs_f]
        cs_f_i = [int(np.imag(c_i)) for c_i in cs_f]

        # recreate complex number and cast to an array
        cs_f = [complex(c_r, c_i) for c_r, c_i in zip(cs_f_r, cs_f_i)]
        cs_f = np.array(cs_f)

        # ---------------------------check FFT (win and no win)----------------------------------- 
        center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]

        if i==1 and do_plots:
            plt.semilogy(center_freqs[1:nFFT//2], np.abs(cs_f[1:nFFT//2]), '-r')
            plt.title('FFT Spectrum -- first 1024')
            plt.show()
            plt.close()
        
        if i==1:
            with open('channel' + str(ci+1) + '_fft_real.txt', 'w') as output:
                for c_r in cs_f_r:
                    output.write(format(np.int32(c_r) & 0xffffffff, '08X') + '\n')
            
            with open('channel' + str(ci+1) + '_fft_real.txt', 'w') as output:
                for c_i in cs_f_i:
                    output.write(format(np.int32(c_i) & 0xffffffff, '08X') + '\n')
        # ---------------------------------------------------------------------------
        
        # save it
        c_fd.append(cs_f)

    # save the output for each channel - vector of 1024-pt FFTs
    channels_fd.append(c_fd) 

# ------------------------------------------------------------------------------------

# ------------------------------ Power Calculation -----------------------------------
# QUESTION: c^2 in FPGA for bfield?
spectra = []
xspectra = []

# loop through the channels
for ci, c in enumerate(channels_fd): 
    spectra.append(power_spectra(c))

    # --------------------------check power calc-------------------------------------
    if do_plots:
        plt.semilogy(center_freqs[1:nFFT//2], power_spectra(c)[1][1:nFFT//2])
        plt.title('Power Spectra first FFT')
        plt.show()
        plt.close()
  
    with open('channel'+str(ci+1)+'_spectra.txt', 'w') as output:
        for s in power_spectra(c)[0]:
            output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')

  # -------------------------------------------------------------------------------

# loop through the channels and perform xspectra calcs
for i in range(0,len(channels_fd)):
    for j in range(i+1,len(channels_fd)):
        Preal, Pimag = power_xspectra(channels_fd[i], channels_fd[j])
        xspectra.append(Preal)
        xspectra.append(Pimag)

# ---------------------------------- check output ------------------------------------
with open('channel_xspectra_real.txt', 'w') as output:
    for s in xspectra[0][0]:
        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
with open('channel_xspectra_imag.txt', 'w') as output:
    for s in xspectra[1][0]:
        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
# ------------------------------------------------------------------------------------  

# -------------------------------------- Time Avg ------------------------------------
# time average for each spectra and cross spectra
nsec = 1 # len of time to accumulate for
spectra_tavg = [time_avg(s, nFFT, fs, nsec) for s in spectra]
xspectra_tavg = [time_avg(xs, nFFT, fs, nsec) for xs in xspectra]

# need to convert to int here! (floor)

if do_plots:
    for st in spectra_tavg:
        plt.semilogy(center_freqs[1:nFFT//2], st[0][1:nFFT//2])
        plt.title('After averaging in time - first second')
        plt.show()
        plt.close()
# ------------------------------------------------------------------------------------ 


# ---------------------------- Rebin into CANVAS bins ---------------------------------
# parse text file with canvas bins
fname = 'fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]

# parse text file with VLF TX canvas bins
fname = 'tx_fbins.txt'                                 
TX_fbins_str = np.genfromtxt(fname, dtype='str') 
TX_fbins_cen = [(float(f[3:].replace(',','')))*1e3 for f in TX_fbins_str]
TX_fbins_names = [TXn[:3] for TXn in TX_fbins_str]
TX_fbins_dbl = [(f - 100., f + 100.) for f in TX_fbins_cen]

# monotonic and 1D list of canvas fbins
c_fbins = [item for sublist in fbins_dbl for item in sublist]
tx_fbins = [item for sublist in TX_fbins_dbl for item in sublist]

# rebin to get average for canvas fbins from the time averaged spectra and xspecrta
spectra_favg = [rebin_canvas(s, c_fbins, center_freqs) for s in spectra_tavg]
xspectra_favg = [rebin_canvas(xs, c_fbins, center_freqs) for xs in xspectra_tavg]

# rebin to get average for canvas VLF TX fbins from the time averaged spectra and xspecrta
spectra_tx_favg = [rebin_canvas(s, tx_fbins, center_freqs) for s in spectra_tavg]
xspectra_tx_favg = [rebin_canvas(xs, tx_fbins, center_freqs) for xs in xspectra_tavg]

# QUESTION HOW TO ADD IN THE END??

# parse text file with center canvas bins
fname = 'fbins_center.txt'                                 
fbins_c_str = np.genfromtxt(fname, dtype='str') 
fbins_center = [(float(f.replace(',',''))) for f in fbins_c_str]

# ------------------------------------------------------------------------------------ 
if do_plots:
    for ft, ft_tx in zip(spectra_favg, spectra_tx_favg):
        plt.semilogy(fbins_center, ft[0][0:nFFT//2])
        plt.semilogy(TX_fbins_cen, ft_tx[0][0:nFFT//2])
        plt.title('After averaging bins - first second')
        plt.show()
        plt.close()
# ------------------------------------------------------------------------------------


# -------------------------------- Compression ---------------------------------------
# log 2 compression -- compressing 64 bit unsigned integer - 6 int and 6 fractional 
spectra_compressed = [np.log2(sc) * 64 for sc in spectra_favg]

# need to extract the sign, save it, compress, and put back the sign after decompression
xspectra_sign = [np.sign(xsc) for xsc in xspectra_favg]
xspectra_compressed = [np.log2(np.abs(xsc)) * 64 for xsc in xspectra_favg]
# HOW TO REMOVE LAST VALUE? -- do we loose a fractional or int bit? double check 
# take the fractional part, multiply by 64 (spectra) and 32 (xspectra), then floor


# repeat for no averaging

# log 2 compression -- compressing 64 bit unsigned integer - 6 int and 6 fractional 
spectra_compressed_single = [np.log2(sc) * 64 for sc in spectra[:][0]]

# need to extract the sign, save it, compress, and put back the sign after decompression
xspectra_sign_single = [np.sign(xsc) for xsc in xspectra[:][0]]
xspectra_compressed_single = [np.log2(np.abs(xsc)) * 64 for xsc in xspectra[:][0]]

# take the fractional part, multiply by 64 (spectra) and 32 (xspectra), then floor
# ------------------------------------------------------------------------------------

# -------------------------------- Decompression -------------------------------------
spectra_dc = [2**(sc / 64) for sc in spectra_compressed]
# include the sign back here
xspectra_dc_nosign = [2**(xsc / 64) for xsc in xspectra_compressed] # SHOULD IT BE 64?
xspectra_dc = [x_sign * xsc for x_sign, xsc in zip(xspectra_sign, xspectra_dc_nosign)]
# ------------------------------------------------------------------------------------


# -------------------------------- Spectrogram ---------------------------------------
t_size = np.shape(spectra_favg)[1]    # second dimension is number of time points
tt = np.arange(1,t_size+1,1)          # create a time vector

fig, axs = plt.subplots(1, len(spectra))
plt.subplots_adjust(wspace=0.3,hspace=0.5)

for i, s in enumerate(spectra_dc):
    s = np.array(s).T
    pcm = axs[i].pcolormesh(tt, fbins_center, np.log10(s), cmap = plt.cm.jet)
    axs[i].set_title('channel'+ str(i))

# set labels
fig.colorbar(pcm, ax=axs[len(spectra_dc)-1])
axs[0].set_ylabel('freq [Hz]')
axs[0].set_xlabel('time [s]')
axs[1].set_xlabel('time [s]')
#plt.show()
plt.close()

# ------------------------------------------------------------------------------------