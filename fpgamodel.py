import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from readFPGA import read_FPGA_input, read_INT_input, quick_compare
from inputstimulus import test_signal
from win import get_win
from fftcanvas import canvas_fft
from fftpwr import fft_spec_power
from rebinacc import rebin_likefpga, acc_likefpga
from cfbinavg import rebin_canvas

fs = 131072.               # sampling freq. 
sample_len = 1024*5/fs    # seconds
signal_freq1 = 8.3e3       # signal freq. 1
#signal_freq2 = 14.7e3     # signal freq. 2
amp = 249.                 # amplitudes
shift = 0.                 # shift

nFFT = 1024
n_acc = 8

# remove output files in path
files = glob.glob('output/*')
for f in files:
    os.remove(f)

# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
channels_td = test_signal(fs, sample_len, [signal_freq1], [amp], show_plots=False, save_output=None)

fpga_in = read_FPGA_input('FPGA/sample_input.txt', 16, signed=True, show_plots=False)
channels_td = [fpga_in]

# STEP 2 ----------------- GET HANNING WINDOW ----------------------------
win = get_win(nFFT, show_plots=False, save_output=None)

# STEP 3 ----------------------- TAKE FFT --------------------------------
channels_fd_real, channels_fd_imag = canvas_fft(nFFT, fs, win, channels_td, overlap=True, show_plots=False, save_output='both')

# or get fft from FPGA -- 
fpga_r = read_FPGA_input('FPGA/fbin_fft_real.txt', 32, signed=True, show_plots=False)
fpga_i = read_FPGA_input('FPGA/fbin_fft_imgry.txt', 32, signed=True, show_plots=False)

channels_fd_real = [[fpga_r[i:i+512] for i in range(0,len(fpga_r),512)]]
channels_fd_imag = [[fpga_i[i:i+512] for i in range(0,len(fpga_r),512)]]

# STEP 4 ----------------------- CALC PWR --------------------------------
spec_pwr = fft_spec_power(channels_fd_real, channels_fd_imag, show_plots=False, save_output='both')

# STEP 5 -------------------- rebin and acc -------------------------------
rebin_pwr = rebin_likefpga(spec_pwr, show_plots=False, save_output='both')
acc_pwr = acc_likefpga(rebin_pwr, n_acc, show_plots=False, save_output='both')

# STEP 5 ---------------- average in time and freq -------------------------
fname = 'CANVAS_fbins/fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]
c_fbins = [item for sublist in fbins_dbl for item in sublist]
center_freqs = [fs/nFFT * ff for ff in np.arange(0, 512)]

avg_pwr = rebin_canvas(acc_pwr, n_acc, c_fbins, center_freqs, show_plots=False, save_output='both')

# STEP 6 ------------------------ compress ---------------------------------
fpga_p = read_FPGA_input('FPGA/log2_output.txt', 8, signed=False, show_plots=False)

# need to clean up the plots and a few other functions