import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_input_2line, read_FPGA_fft_debug, read_FPGA_input_7line
from inputstimulus import test_signal, input_chirp, white_noise
from win import get_win
from fftcanvas import canvas_fft
from fftpwr import fft_spec_power, fft_xspec_power
from rebinacc import rebin_likefpga, acc_likefpga
from cfbinavg import rebin_canvas


# remove output files in path
files = glob.glob('output/*')
for f in files:
    os.remove(f)
files = glob.glob('plots_1_20/*')
for f in files:
    os.remove(f)
 
#freqs = [428, 967, 1.73e3, 4.12e3, 8.57e3, 11.92e3, 17.41e3, 23.56e3, 38.97e3]
#shifts = [57*np.pi/(2*59), 63*np.pi/67, 3*41*np.pi/(2*43), 2*16*np.pi/19]
#str_freqs =  ['428', '967', '1.73k', '4.12k', '8.57k', '11.92k', '17.41k', '23.56k', '38.97k']

#for str_f, signal_freq1 in zip(str_freqs, freqs):
#    for shift in shifts:
fs = 131072.               # sampling freq. 
sample_len = 1024*20/fs    # seconds
signal_freq1 = 33e3      # signal freq. 1
#signal_freq2 = 14.7e3     # signal freq. 2
amp = 2**15-5423           # amplitudes
shift =0.  # shift

#ss_str = str(round(shift,2))
#s_str = ss_str[0]+'p'+ss_str[2:]

nFFT = 1024
n_acc = 8

# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
#channels_td = test_signal(fs, sample_len, [signal_freq1], [amp], shift=shift, show_plots=False, save_output='hex')
#channels_td = white_noise(fs, 1024*20/fs, 2**15-7985, show_plots=True, save_output='hex')
#channels_td = input_chirp(fs, 1024*20/fs, 19.52e3, 567, 2039, show_plots=True, save_output='hex')

#fpga_in = read_FPGA_input('FPGA/channel0_input_1.73_1.52_hex.txt', 16, signed=True, show_plots=True)
#diff = quick_compare(channels_td[0], fpga_in, 1024, 'input', show_plots=True)

# STEP 2 ----------------- GET HANNING WINDOW ----------------------------
#win = get_win(nFFT, show_plots=False, save_output=None)

# STEP 3 ----------------------- TAKE FFT --------------------------------
#channels_fd_real, channels_fd_imag = canvas_fft(nFFT, fs, win, channels_td, overlap=True, show_plots=False, save_output=None)

# or get fft from file
#fpga_ri = read_FPGA_input_2line('FPGA/fft_real_imag_33khz_fix.txt', 32, signed=True, show_plots=False)

#fpga_i = [fpga_ri[n] for n in range(0,len(fpga_ri),2)]
#fpga_r = [fpga_ri[n] for n in range(1,len(fpga_ri),2)]

#channels_fd_realf = [[fpga_r[i:i+512] for i in range(0,len(fpga_r),512)]]
#channels_fd_imagf = [[fpga_i[i:i+512] for i in range(0,len(fpga_r),512)]]

#py_r = flatten(channels_fd_real[0])
#py_i = flatten(channels_fd_imag[0])

#quick_compare(py_r[:38*512], fpga_r[:38*512], 512, 'ffimag_512Hz', show_plots=True)

ch0_fft_imag = read_FPGA_input('FPGA/channel0_fft_imag_hex.txt',32,signed=True, show_plots=False)
ch0_fft_real = read_FPGA_input('FPGA/channel0_fft_real_hex.txt',32,signed=True, show_plots=False)
ch1_fft_imag = read_FPGA_input('FPGA/channel1_fft_imag_hex.txt',32,signed=True, show_plots=False)
ch1_fft_real = read_FPGA_input('FPGA/channel1_fft_real_hex.txt',32,signed=True, show_plots=False)

# STEP 4 ----------------------- CALC PWR --------------------------------
xspec_pwr_r, xspec_pwr_i = fft_xspec_power(ch0_fft_real, ch0_fft_imag, ch1_fft_real, ch1_fft_imag)
xspec_r_f, xspec_i_f = read_FPGA_input_7line('FPGA/xspec_fbin_pwr.txt',64,signed=True,show_plots=False)
print(xspec_i_f[:10], xspec_pwr_i[:10])
quick_compare(xspec_pwr_i[:len(xspec_i_f)], xspec_i_f,18000,show_plots=True)

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

py_array = np.array(flatten(flatten(avg_pwr)))

#quick_compare(py_array, fp_array, len(py_array), 'after avg')
"""
datalines = [line.strip() for line in f]
output_val = [int(line[-3:],16) for line in datalines]
input_val = [int(line[:-4].strip(),16) for line in datalines]

input_val = input_val[1:]
output_val = output_val[1:]

cmprs_val = [np.ceil(math.log2(iv)*64) for iv in input_val]

diff = quick_compare(output_val, cmprs_val, len(cmprs_val)//10, 'ceil_iosweep', show_plots=True)
"""