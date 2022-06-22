from cProfile import label
from pyexpat import model
from readFPGA import read_FPGA_fft, read_FPGA_input, read_FPGA_input_lines
import numpy as np
import matplotlib.pyplot as plt

show_plots = False
include_transmitter = True

file_path = "./Data_compare/x-spec/"

amp = "hi_amp_" #valid options are hi_amp, low_amp, and mid_amp

simulation_file = file_path+f+'_fft_fbin_pwr.txt'

#Base File paths
pypy_base = file_path+'Python_Results/'+'pypy_'+amp
FPGA_base = file_path+"FPGA_Results/FPGA-_"+amp

freqs = 3
phases = 3
if include_transmitter:
    freqs+=1
results = np.empty([freqs, phases])
results = np.NaN

for f in range(freqs):
    #Set the frequency part of the label
    if f == 0:
        freq = '03kHz'
    elif f == 1:
        freq = '10kHz'
    elif f == 2:
        freq = '24kHz'
    else:
        print("Unknown Frequency")
    for ph in range(phases):
        #Set the phase part of the label
        if ph == 0:
            phase = '5'
        elif ph == 1:
            phase = '35'
        elif ph == 2:
            phase = '83'
        else:
            print("Unknown Phase")

        test_conditions = freq+phase

        #Read in files and set dtype appropriately 
        fpga_comp,fpga_avg = read_FPGA_input_lines(fpga_file_res,64,line_n=3,x=1,y=2,signed=False)
        fpga_avg = np.array(fpga_avg,dtype=np.int64)
        fpga_comp = np.array(fpga_comp,dtype=np.int32)

        pypy_avg =  np.array(read_FPGA_input(pypy_file_avg,64,False,show_plots=False),dtype=np.int64)
        pypy_comp =  np.array(read_FPGA_input(pypy_file_comp,16,False,show_plots=False),dtype=np.int32)

        #Set the center bin for each frequency
        transmitter = False
        if freq == '60khz':
            center = 56
        elif freq == '33khz':
            center = 53
        elif freq == '24khz':
            center = 48
            transmitter = True
            t_bin = 61
        elif freq == '10khz':
            center= 37
        elif freq == '03khz':
            center = 19
        elif freq == '512hz':
            center = 2

        #calculate deltas
        pypy_avg_delta = pypy_avg[center] - fpga_avg[center]
        pypy_comp_delta = pypy_comp[center] - fpga_comp[center]

        if transmitter:
            pypy_avg_delta_T = pypy_avg[t_bin] - fpga_avg[t_bin]
            pypy_comp_delta_T = pypy_comp[t_bin] - fpga_comp[t_bin]
        
        #normalize to python model results
        FPGA_pypy_avg_compare = (pypy_avg_delta)/pypy_avg[center]
        FPGA_pypy_comp_compare = (pypy_comp_delta )/pypy_comp[center]

        if transmitter:
            FPGA_pypy_avg_compare_T = (pypy_avg_delta)/pypy_avg[t_bin]
            FPGA_pypy_comp_compare_T = (pypy_comp_delta )/pypy_comp[t_bin]

        #put the result in the results matrix
        #results[f][ph] = FPGA_pypy_avg_compare
        results[f][ph] = FPGA_pypy_comp_compare
        if transmitter:
            #use the last freq bin as the 24 khz transmitter bin
            #results[freqs][ph] = FPGA_pypy_avg_compare_T
            results[freqs][ph] = FPGA_pypy_comp_compare_T

print(results)
print("done")