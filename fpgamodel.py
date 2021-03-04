import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_fft_debug, read_FPGA_input_lines
from inputstimulus import test_signal, input_chirp, white_noise
from win import get_win
from fftcanvas import canvas_fft
from fftpwr import fft_spec_power, fft_xspec_power
from rebinacc import rebin_likefpga, acc_likefpga
from cfbinavg import rebin_canvas, fix_neg1
from log2compress import spec_compress, xspec_compress

from saveas import saveascsv

# remove output files in path
files = glob.glob('output/*')
for f in files:
    os.remove(f)

# some set up parameters

fs = 131072.             # sampling freq. 
signal_freq1 = 35e3    # signal freq. 1
signal_freq2 = 35e3 # signal freq. 2
amp = 2**15         # amplitudes
shift = 0  # shift

period = 1/signal_freq1
sample_len = 64*1024/fs # seconds
sample_len = 1024*256*2/fs
nFFT = 1024
n_acc = 256

# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
channels1_td = test_signal(fs, sample_len, signal_freq1, amp, 1, shift=0, show_plots=False, save_output='both')
channels2_td = test_signal(fs, sample_len, signal_freq2, amp, 1, shift=shift, show_plots=False, save_output='both')

# STEP 2 ----------------- GET HANNING WINDOW ----------------------------
win = get_win(nFFT, show_plots=False, save_output=None)

# STEP 3 ----------------------- TAKE FFT --------------------------------
channel1_fd_real, channel1_fd_imag = canvas_fft(nFFT, fs, win, channels1_td, 0, overlap=True, show_plots=False, save_output='both')
channel2_fd_real, channel2_fd_imag = canvas_fft(nFFT, fs, win, channels2_td, 0, overlap=True, show_plots=False, save_output='both')

print(max(channel1_fd_real)-2**29,max(channel1_fd_imag)-2**29)
print(max(channel2_fd_real)-2**29,max(channel2_fd_imag)-2**29)

#f_ar, f_ai = read_FPGA_input_lines('FPGA/stim_fft_data.txt', 32, 4, 0, 1)
#f_br, f_bi = read_FPGA_input_lines('FPGA/stim_fft_data.txt', 32, 4, 2, 3)

# STEP 4 ----------------------- CALC PWR --------------------------------
#spec_pwr = fft_spec_power(channel1_fd_real,channel1_fd_imag)

xspec_r, xspec_i = fft_xspec_power(channel1_fd_real, channel1_fd_imag, channel2_fd_real, channel2_fd_imag)
#xspec_r_f, xspec_i_f = read_FPGA_input_lines('FPGA/fft_fbin_pwr.txt', 64, 7, 5, 6, signed=True)
print(max(np.abs(xspec_r))-2**58,max(np.abs(xspec_i))-2**58)
#quick_compare(xspec_r, xspec_r_f, len(xspec_r), show_plots=True)
"""
# STEP 5 -------------------- rebin and acc -------------------------------
rebin_pwr_r = rebin_likefpga(xspec_r, 0, show_plots=False, save_output='both')
rebin_pwr_i = rebin_likefpga(xspec_i, 0, show_plots=False, save_output='both')

#acc_pwr_r = acc_likefpga(rebin_pwr_r, n_acc, 0, show_plots=False, save_output='both')
#acc_pwr_i = acc_likefpga(rebin_pwr_i, n_acc, 0, show_plots=False, save_output='both')

acc_f_r, acc_f_i = read_FPGA_input_lines('FPGA/fbin_total_pwr.txt', 64, 3, 1, 2, signed=True)
#quick_compare(acc_pwr_r, acc_f_r, 330*5, show_plots=True)


rebin_pwr= rebin_likefpga(spec_pwr, 0, show_plots=False, save_output='both')
acc_pwr = acc_likefpga(rebin_pwr, n_acc, 0, show_plots=False, save_output='both')
mysum = 0
for i in range(0,330*256,330):
    mysum+=int(rebin_pwr[i])

print(mysum)
print(max(acc_pwr))

# STEP 6 ---------------- average in time and freq -------------------------
fname = 'CANVAS_fbins/fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]
c_fbins = [item for sublist in fbins_dbl for item in sublist]
center_freqs = [fs/nFFT * ff for ff in np.arange(0, 512)]

avg_pwr = rebin_canvas(acc_pwr, n_acc, c_fbins, center_freqs,0,tx_bins=True,show_plots=False, save_output='both')

#avg_pwr_r = rebin_canvas(acc_pwr_r, n_acc, c_fbins, center_freqs,0,tx_bins=True,show_plots=False, save_output='both')
#avg_pwr_i = rebin_canvas(acc_pwr_i, n_acc, c_fbins, center_freqs,0,tx_bins=True,show_plots=False, save_output='both')

#f_avg_r, f_compress_r = read_FPGA_input_lines('FPGA/real_bin_avg_pwr.txt', 64, 3, 1, 2, signed=True)
#f_avg_i, f_compress_i = read_FPGA_input_lines('FPGA/imgy_bin_avg_pwr.txt', 64, 3, 1, 2, signed=True)

#avg_pwr_r_fix = fix_neg1(avg_pwr_r, f_avg_r)
#avg_pwr_i_fix = fix_neg1(avg_pwr_i, f_avg_i)

#quick_compare(f_avg_i, avg_pwr_i_fix, 67*5, show_plots=True)

# STEP 7 ---------------- compress -------------------------

cmprs_val_r = xspec_compress(avg_pwr_r_fix,0)
cmprs_val_i = xspec_compress(avg_pwr_i_fix,0)

#diff = quick_compare(cmprs_val_i, f_compress_i, 67*5, show_plots=True)
f_avg, f_compress = read_FPGA_input_lines('FPGA/bin_avg_pwr.txt', 64, 3, 1, 2, signed=True)

cmprs_val = spec_compress(f_avg,0)
#diff = quick_compare(cmprs_val, f_compress, 67*10, show_plots=True)
print(cmprs_val[195])
print(f_compress[195])
print(f_avg[195])
"""