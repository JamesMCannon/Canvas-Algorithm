from cProfile import label
from pyexpat import model
from readFPGA import read_FPGA_fft, read_FPGA_input, read_FPGA_input_lines, read_FPGA_xspectra, read_FPGA_cmprs
import numpy as np
import matplotlib.pyplot as plt

show_plots = False
include_transmitter = True

file_path = "./Data_compare/x-spec/"

amp = "high-high_" #valid options are hi_amp, low_amp, and mid_amp
rev = '60220713_'

#Base File paths
pypy_base = file_path+'Python_Results/'+amp
FPGA_base = file_path+"FPGA_Results/FPGA-"+rev+amp

freqs = 3
phases = 3
if include_transmitter:
    freqs+=1
results_im = np.empty([freqs, phases])
results_im[:][:] = np.NaN

results_re = np.empty([freqs, phases])
results_re[:][:] = np.NaN

for f in range(freqs):
    #Set the frequency part of the label
    if f == 0:
        freq = '03kHz_'
    elif f == 1:
        freq = '10kHz_'
    elif f == 2:
        freq = '24kHz_'
    else:
        print("Unknown Frequency")
    for ph in range(phases):
        #Set the phase part of the label
        if ph == 0:
            phase = '5deg_'
        elif ph == 1:
            phase = '35deg_'
        elif ph == 2:
            phase = '83deg_'
        else:
            print("Unknown Phase")

        #assign file names
        test_conditions = phase+freq

        fpga_file_im = FPGA_base+test_conditions+'xpec_im_avg_hex.txt'
        fpga_file_re = FPGA_base+test_conditions+'xpec_re_avg_hex.txt'

        pypy_file_avg_im = pypy_base+test_conditions+'i_avg_hex.txt'
        pypy_file_avg_re = pypy_base+test_conditions+'r_avg_hex.txt'

        pypy_file_comp_im = pypy_base+test_conditions+'i_cmprs_hex.txt'
        pypy_file_comp_re = pypy_base+test_conditions+'r_cmprs_hex.txt'

        #Read in files
        fpga_comp_im,fpga_avg_im = read_FPGA_xspectra(fpga_file_im,line_n=1)
        fpga_comp_re,fpga_avg_re = read_FPGA_xspectra(fpga_file_re,line_n=1)

        pypy_avg_im =  read_FPGA_input(pypy_file_avg_im,64,True,show_plots=False)
        pypy_avg_re =  read_FPGA_input(pypy_file_avg_re,64,True,show_plots=False)

        pypy_comp_im =  read_FPGA_cmprs(pypy_file_comp_im,line_n=0)
        pypy_comp_re =  read_FPGA_cmprs(pypy_file_comp_re,line_n=0)

        #Set the center bin for each frequency
        transmitter = False
        if freq == '60kHz_':
            center = 56
        elif freq == '33kHz_':
            center = 53
        elif freq == '24kHz_':
            center = 48
            transmitter = True
            t_bin = 61
        elif freq == '10kHz_':
            center= 37
        elif freq == '03kHz_':
            center = 19
        elif freq == '512Hz_':
            center = 2
        else:
            print('invalid frequency')

        #calculate deltas
        pypy_avg_delta_im = pypy_avg_im[center] - fpga_avg_im[center]
        pypy_comp_delta_im = pypy_comp_im[center] - fpga_comp_im[center]

        if transmitter:
            pypy_avg_delta_T_im = pypy_avg_im[t_bin] - fpga_avg_im[t_bin]
            pypy_comp_delta_T_im = pypy_comp_im[t_bin] - fpga_comp_im[t_bin]
        
        pypy_avg_delta_re = pypy_avg_re[center] - fpga_avg_re[center]
        pypy_comp_delta_re = pypy_comp_re[center] - fpga_comp_re[center]

        if transmitter:
            pypy_avg_delta_T_re = pypy_avg_re[t_bin] - fpga_avg_re[t_bin]
            pypy_comp_delta_T_re = pypy_comp_re[t_bin] - fpga_comp_re[t_bin]
        
        #normalize to python model results
        FPGA_pypy_avg_compare_im = (pypy_avg_delta_im)/pypy_avg_im[center]
        FPGA_pypy_comp_compare_im = (pypy_comp_delta_im )/pypy_comp_im[center]

        if transmitter:
            FPGA_pypy_avg_compare_T_im = (pypy_avg_delta_im)/pypy_avg_im[t_bin]
            FPGA_pypy_comp_compare_T_im = (pypy_comp_delta_im )/pypy_comp_im[t_bin]
        
        FPGA_pypy_avg_compare_re = (pypy_avg_delta_re)/pypy_avg_re[center]
        FPGA_pypy_comp_compare_re = (pypy_comp_delta_re )/pypy_comp_re[center]

        if transmitter:
            FPGA_pypy_avg_compare_T_re = (pypy_avg_delta_re)/pypy_avg_re[t_bin]
            FPGA_pypy_comp_compare_T_re = (pypy_comp_delta_re )/pypy_comp_re[t_bin]

        #put the result in the results matrix
        #results[f][ph] = FPGA_pypy_avg_compare
        results_im[int(f)][int(ph)] = FPGA_pypy_comp_compare_im
        if transmitter:
            #use the last freq bin as the 24 khz transmitter bin
            #results[freqs-1][ph] = FPGA_pypy_avg_compare_T
            results_im[freqs-1][ph] = FPGA_pypy_comp_compare_T_im
        
        results_re[int(f)][int(ph)] = FPGA_pypy_comp_compare_re
        if transmitter:
            #use the last freq bin as the 24 khz transmitter bin
            #results[freqs-1][ph] = FPGA_pypy_avg_compare_T
            results_re[freqs-1][ph] = FPGA_pypy_comp_compare_T_re

print('Imaginary Results: \n',results_im)
print('Real Results: \n',results_re)
print("done")