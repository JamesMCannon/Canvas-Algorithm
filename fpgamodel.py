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
n_acc = 8

# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
#channels1_td = test_signal(fs, sample_len, signal_freq1, amp, 1, shift=0, show_plots=False, save_output='both')

# or input from a file
channels1_td = read_FPGA_input('FPGA/497_32p8_494_8165_s1piover7_input_hex.txt',16, show_plots=False)

# STEP 2 ----------------- GET HANNING WINDOW ----------------------------
win = get_win(nFFT, show_plots=False, save_output=None)

# STEP 3 ----------------------- TAKE FFT --------------------------------
channel1_fd_real, channel1_fd_imag = canvas_fft(nFFT, fs, win, channels1_td, 0, overlap=True, show_plots=False, save_output='both')
f_ar, f_ai = read_FPGA_input_lines('FPGA/fft_fbin_pwr.txt', 32, 4, 1, 2)
print(channel1_fd_imag[254:258],f_ai[254:258])

for i, (fi, pu) in enumerate(zip(f_ai,channel1_fd_imag)):
    if pu!=0:
        dd = (fi - pu) / pu
        if np.abs(dd) > 0.5 and np.abs(pu) > 500:
            print(i,fi,pu)

# STEP 4 ----------------------- CALC PWR --------------------------------
spec_pwr = fft_spec_power(f_ar, f_ai)
spc, fspec = read_FPGA_input_lines('FPGA/fft_fbin_pwr.txt', 64, 4, 2, 3)

#quick_compare(fspec,spec_pwr,len(spec_pwr),show_plots=True)

# STEP 5 -------------------- rebin and acc -------------------------------
spc, fspec = read_FPGA_input_lines('FPGA/fft_fbin_pwr.txt', 64, 4, 2, 3)

rebin_pwr= rebin_likefpga(spec_pwr, 0, show_plots=False, save_output='both')
acc_pwr = acc_likefpga(rebin_pwr, n_acc, 0, show_plots=False, save_output='both')


# STEP 6 ---------------- average in time and freq -------------------------
# import canvas bins correctly
fname = 'CANVAS_fbins/fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]
c_fbins = [item for sublist in fbins_dbl for item in sublist]
center_freqs = [fs/nFFT * ff for ff in np.arange(0, 512)]

avg_pwr = rebin_canvas(acc_pwr, n_acc, c_fbins, center_freqs,0,tx_bins=True,show_plots=False, save_output='both')

bavg, cpmrs = read_FPGA_input_lines('FPGA/bin_avg_pwr.txt',64,3,1,2)

quick_compare(bavg,avg_pwr,1024,show_plots=True)

# STEP 7 ---------------- compress -------------------------
cmprs_val = spec_compress(avg_pwr,0)
quick_compare(cmprs_val,cpmrs,1024,show_plots=True)
