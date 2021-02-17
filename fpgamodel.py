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
from cfbinavg import rebin_canvas

from saveas import saveascsv

# remove output files in path
#files = glob.glob('output/*')
#for f in files:
#    os.remove(f)

# TODO clean up documentation and save output better!! No channel segments 

fs = 131072.               # sampling freq. 
#sample_len = 1024*20/fs    # seconds
#signal_freq1 = 33e3      # signal freq. 1
#signal_freq2 = 14.7e3     # signal freq. 2
#amp = 2**15-5423           # amplitudes
#shift =0.  # shift

nFFT = 1024
n_acc = 8
ftest = '60k'
# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
#channels_td = test_signal(fs, sample_len, signal_freq1, amp, 1, shift=shift, show_plots=False, save_output='both')
#channels_td = white_noise(fs, 1024*20/fs, 2**15-7985, show_plots=True, save_output='hex')
#channels_td = input_chirp(fs, 1024*20/fs, 19.52e3, 567, 2039, show_plots=True, save_output='hex')

channel1_td = read_FPGA_input('FPGA/stim_adc_data_'+ftest+'hz_180ms.txt', 16, signed=True, show_plots=False)

# STEP 2 ----------------- GET HANNING WINDOW ----------------------------
win = get_win(nFFT, show_plots=False, save_output=None)

# STEP 3 ----------------------- TAKE FFT --------------------------------
channel1_fd_real, channel1_fd_imag = canvas_fft(nFFT, fs, win, channel1_td, 1, overlap=True, show_plots=False, save_output=None)
f_r, f_i = read_FPGA_input_lines('FPGA/fft_fbin_pwr_'+ftest+'hz.txt', 32, 4, 1, 2)

adds = []
diff = quick_compare(channel1_fd_real, f_r, 512*44, show_plots=False)

for ind, (fr, cr, dd) in enumerate(zip(f_r[:512*44], channel1_fd_real[:512*44],diff)):
    add=[ind, fr, cr, dd]
    adds.append(add)

saveascsv(ftest+'hz.csv',adds)

# STEP 4 ----------------------- CALC PWR --------------------------------
spec_pwr = fft_spec_power(channel1_fd_real, channel1_fd_imag)
spec_pwr_f = fft_spec_power(f_r, f_i)
s_discard, s = read_FPGA_input_lines('FPGA/fft_fbin_pwr_'+ftest+'hz.txt', 64, 4, 0, 3)
#diff = quick_compare(spec_pwr_f[:512*44], s[:512*44], 512*44, show_plots=True)

# STEP 5 -------------------- rebin and acc -------------------------------
rebin_pwr_r = rebin_likefpga(spec_pwr_f, 0, show_plots=False, save_output='both')
acc_pwr_r = acc_likefpga(rebin_pwr_r, n_acc,'real', show_plots=False, save_output='both')

# acc is wrong, use this
s_discard, sacc = read_FPGA_input_lines('FPGA/fbin_total_pwr_'+ftest+'hz.txt', 64, 2, 0, 1,signed=False)

acc_f2 = np.zeros((len(rebin_pwr_r)//(330*n_acc),330))
for i in range(0,len(rebin_pwr_r)//(330*n_acc)):
    
    for k in range(330):
        argh = []
        for j in range(8*i*330,8*(i+1)*330,330):
            check = rebin_pwr_r[j:j+330]
            argh.append(check[k])
        thatval = sum(argh)
        acc_f2[i][k] = thatval

quick_compare(flatten(acc_f2), sacc, 330*5, show_plots=True)
vals = 330*44//8
#diff = quick_compare(acc_pwr_r[:vals], sacc[:vals], vals, show_plots=True)

"""
# STEP 5 ---------------- average in time and freq -------------------------
fname = 'CANVAS_fbins/fbins.txt'                                 
fbins_str = np.genfromtxt(fname, dtype='str') 
fbins_dbl = [(float(f[0].replace(',','')),float(f[1].replace(',',''))) for f in fbins_str]
c_fbins = [item for sublist in fbins_dbl for item in sublist]
center_freqs = [fs/nFFT * ff for ff in np.arange(0, 512)]

avg_pwr_r = rebin_canvas(acc_pwr_r, n_acc,'real', c_fbins, center_freqs, show_plots=False, save_output='both')
vals = 330*44//8
s_discard, sacc = read_FPGA_input_lines('FPGA/fbin_total_pwr_'+ftest+'hz.txt', 64, 2, 0, 1)
quick_compare(acc_pwr_r[:vals], sacc[:vals], vals, show_plots=True)



datalines = [line.strip() for line in f]
output_val = [int(line[-3:],16) for line in datalines]
input_val = [int(line[:-4].strip(),16) for line in datalines]

input_val = input_val[1:]
output_val = output_val[1:]

cmprs_val = [np.ceil(math.log2(iv)*64) for iv in input_val]

diff = quick_compare(output_val, cmprs_val, len(cmprs_val)//10, 'ceil_iosweep', show_plots=True)
"""