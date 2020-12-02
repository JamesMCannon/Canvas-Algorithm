# import statements
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from scipy.fftpack import fft, fftfreq, fftshift
from canvas_alg_helper_funcs import get_vlfdata, resample, get_win, power_spectra, rebin_canvas, power_xspectra, time_avg, twos_complement  

# ----------------------------- Create a Test Signal ------------------------------- 
# create time domain data
fs = 131072.                           # sampling freq. 
sample_len = 10*(2**(-8))               # seconds
t_vec = np.linspace(0, sample_len, num=int(fs*sample_len))   # create time vec
signal_freq1 = 8.3e3                     # signal freq. 1
signal_freq2 = 25.2e3                     # signal freq. 2
amp = 249.                             # signal amplitude -- SIGNED 

# add white noise with spectral density comprable to expected senisitivity for E or B
# leakage in TX bins from 25.2e3 -- look at width of fft limit 

# channels (ex, ey, bx, by, bz)
# shift = np.pi/2  # shift between 2 channels                       

bx = amp * np.sin(signal_freq1 * 2 * np.pi * t_vec)
by = amp * np.sin(signal_freq2 * 2 * np.pi * t_vec)

# make an integer
bx = [round(bxx,0) for bxx in bx]
by = [round(byy,0) for byy in by]

# collect time domain channels here
channels_td = [bx]

do_plots = True
save_output = True

# -----------------------------check input signal------------------------------------
plt_chk = 1024
if do_plots:
    for ch in channels_td:
        plt.plot(t_vec[:plt_chk], ch[:plt_chk])
    plt.title('Input Signal - first 1024')
    plt.show()
    plt.close()

# cast input (ints) to 16bit int represented in hex
if save_output:
    for ci, c in enumerate(channels_td):
        with open('channel'+str(ci)+'_input_hex.txt', 'w') as output:
            for b in c:
                output.write(format(np.int16(b) & 0xffff, '04X') + '\n')
        with open('channel'+str(ci)+'_input_int.txt', 'w') as output:
            for b in c:
                output.write(str(np.int16(b))+'\n')

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
if save_output:
    # cast input (ints) to 16bit int represented in hex
    with open('window.txt', 'w') as output:
        for w in win:
            output.write(str(np.uint16(w))+'\n')
if do_plots:
    plt.plot(np.arange(0, len(win)), win)
    plt.title('Input Window')
    plt.show()
    plt.close()
# ------------------------------------------------------------------------------------

# ---------------------------- Perform FFT -------------------------------------------
# FFT on input * window for every 1024 points shifting by 512 -- 50% overlap 
channels_fd = [] # store channels now in frequency domain

for ci, c in enumerate(channels_td):

    c_fd = [] # store each segment of channel f domain

    # first, handle any channels that aren't 1024*n points in length
    remainder = int(len(c)) % nFFT
    if remainder !=0: # remainder means not an integer multiple of 1024
        need_zero = nFFT - remainder # this will be the # of missing points
        for i in range(need_zero):
            c.append(0)

    # loop through by 512
    for i in range(0, len(c), nFFT//2):
    #for i in range(0, len(c), nFFT): # for no overlap
        cs_2 = c[i:i+nFFT]

        # this is handling the LAST FFT with overlap and padding with 0's
        if len(cs_2) != nFFT:
            cs_2.extend(np.zeros(nFFT//2))

        # mutitply elementwise by windowing func
        cs_2 = np.array(cs_2)
        cs_win = np.multiply(win, cs_2) # should be integer (with max 2^31-1)

        # ---------------------------check win * input---------------------------------
        if save_output:
            with open('channel'+str(ci)+'_win.txt', 'a') as output:
                for csw in cs_win:
                    output.write(str(np.int32(csw))+'\n')

        cs_winw = [np.int32(csw) for csw in cs_win]
        if i==0 and do_plots: # first FFT
            plt.plot(cs_winw)
            plt.title('Input Window x Input Signal - First 1024')
            plt.show()
            plt.close()
        
        # ----------------------------------------------------------------------------

        # take FFT
        cs_f = np.fft.fft(cs_win)

        # make it match IDL
        cs_f = cs_f / nFFT

        # convert real and imag to int
        # only need first half (match IDL/FPGA)
        cs_f_r = [round(np.real(c_r)) for c_r in cs_f[:nFFT//2]]
        cs_f_i = [round(np.imag(c_i)) for c_i in cs_f[:nFFT//2]]

        # recreate complex number and cast to an array
        cs_f = [complex(c_r, c_i) for c_r, c_i in zip(cs_f_r, cs_f_i)]

        # ---------------------------check FFT ----------------------------------- 
        if save_output:
            with open('channel'+str(ci)+'_fft_real_hex.txt', 'a') as output:
                for c_r in cs_f_r:
                    output.write(format(np.int32(c_r) & 0xffffffff, '08X') + '\n')

            with open('channel'+str(ci)+'_fft_imag_hex.txt', 'a') as output:
                for c_i in cs_f_i:
                    output.write(format(np.int32(c_i) & 0xffffffff, '08X') + '\n')
            
            with open('channel'+str(ci)+'_fft_real_int.txt', 'a') as output:
                for c_r in cs_f_r:
                    output.write(str(np.int32(c_r))+'\n')

            with open('channel'+str(ci)+'_fft_imag_int.txt', 'a') as output:
                for c_i in cs_f_i:
                    output.write(str(np.int32(c_i))+'\n')

        center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]
        if i==0 and do_plots:
            plt.semilogy(center_freqs[0:nFFT//2], np.abs(cs_f[0:nFFT//2]), '-r')
            plt.title('FFT')
            plt.show()
            plt.close()
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
    #if do_plots:
    #    plt.semilogy(center_freqs[1:nFFT//2], power_spectra(c)[1][1:nFFT//2])
    #    plt.title('Power Spectra first FFT')
    #    plt.show()
    #    plt.close()
  
    #with open('channel'+str(ci+1)+'_spectra.txt', 'w') as output:
    #    for s in power_spectra(c)[0]:
    #        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
    #    for s in power_spectra(c)[1]:
    #        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')

  # -------------------------------------------------------------------------------

# loop through the channels and perform xspectra calcs
for i in range(0,len(channels_fd)):
    for j in range(i+1,len(channels_fd)):
        Preal, Pimag = power_xspectra(channels_fd[i], channels_fd[j])
        xspectra.append(Preal)
        xspectra.append(Pimag)

# ---------------------------------- check output ------------------------------------
#with open('channel_xspectra_real.txt', 'w') as output:
#    for s in xspectra[0][0]:
#        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
#    for s in xspectra[0][1]:
#        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
#with open('channel_xspectra_imag.txt', 'w') as output:
#    for s in xspectra[1][0]:
#        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
#    for s in xspectra[1][1]:
#        output.write(format(np.uint64(s) & 0xffffffffffffffff, '016X') + '\n')
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
        plt.semilogy(TX_fbins_cen, ft_tx[0][0:nFFT//2], '.') # debug this ?? 
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

for i, s in enumerate(spectra_favg):
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