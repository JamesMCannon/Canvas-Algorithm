from cProfile import label
from pyexpat import model
from readFPGA import read_FPGA_fft, read_FPGA_input, read_FPGA_input_lines
import numpy as np
import matplotlib.pyplot as plt

show_plots = False

file_path = "./Data_compare/"
f = "512hz"
fpga_rev = "Rev14p1"
amp = "hi_amp_" #valid options are hi_amp, low_amp, and mid_amp

simulation_file = file_path+f+'_fft_fbin_pwr.txt'

#Base File paths
pypy_base = file_path+'Python_Results/'+'pypy_'+amp+f
pyFPGA_base = file_path+'Python_Results/'+'pyFPGA_'+amp+f
FPGA_base = file_path+"FPGA_Results/"+fpga_rev+"/FPGA-"+fpga_rev+'_'+amp+f

pypymodel_file_img = pypy_base+'_fft_imag_hex.txt'
pypymodel_file_re = pypy_base+'_fft_real_hex.txt'

fpga_file_FFT = FPGA_base+'_FFT_hex.txt'
fpga_re, fpga_img = read_FPGA_input_lines(fpga_file_FFT, 32, line_n=3, x=1, y=2)

bins = np.arange(512)
freq = bins*128/1000
#sim_re,sim_img =np.array(read_FPGA_fft(simulation_file,32),dtype=np.int64)

model_img = np.array(read_FPGA_input(pypymodel_file_img,32,True,show_plots=False),dtype=np.int64)
model_re = np.array(read_FPGA_input(pypymodel_file_re,32,True,show_plots=False),dtype=np.int64)

fpga_re = fpga_re[:512]
fpga_img = fpga_img[:512]
model_img = model_img[:512]
model_re = model_re[:512]
#sim_img = sim_img[:512]
#sim_re = sim_re[:512]
#plt.style.use('dark_background')
if show_plots:
    plt.plot(freq,model_img[:512],'-',label = 'Python')
    plt.plot(freq,fpga_img[:512],'-',label = 'FPGA')
    plt.plot(freq,sim_img[:512],'-',label = 'Simulation')
    plt.legend()
    plt.title('Imaginary FFT Coefficient')
    plt.xlabel('Frequency (kHz)')
    plt.show()
    plt.close()

    plt.plot(freq,model_re[:512],'-', label = 'Python')
    plt.plot(freq,fpga_re[:512],'-', label='FPGA')
    plt.plot(freq,sim_re[:512],'-',label = 'Simulation')
    plt.legend()
    plt.title('Real FFT Coefficient')   
    plt.xlabel('Frequency (kHz)')
    plt.show()
    plt.close()

#power calcs
pypymodel_file_pwr = pypy_base + '_Spectra_hex.txt'

pypymodel_pwr = np.array(read_FPGA_input(pypymodel_file_pwr,64,signed= False,show_plots=False),dtype=np.uint64)
pypymodel_pwr = pypymodel_pwr[:512]

#pyFPGAmodel_file_pwr = pyFPGA_base + '_Spectra_hex.txt'
#pyFPGAmodel_pwr = np.array(read_FPGA_input(pyFPGAmodel_file_pwr,64,signed= False,show_plots=False),dtype=np.uint64)
#pyFPGAmodel_pwr = pyFPGAmodel_pwr[:512]

#model_pwr = model_re**2 + model_img**2
#sim_pwr = sim_re**2 + sim_img**2

fpga_file_pwr = FPGA_base+'_spectra_hex.txt'
pwr_bin,fpga_pwr = read_FPGA_input_lines(fpga_file_pwr,64,line_n=2,x=0,y=1,signed=False)
fpga_pwr = fpga_pwr[:512]

if show_plots:
    plt.plot(freq,pypymodel_pwr[:512],'-',label='Python')
    plt.plot(freq,fpga_pwr[:512],'-',label='FPGA')
    #plt.plot(freq,sim_pwr[:512],'-',label = 'Simulation')
    plt.title('Signal Power')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Frequency (kHz)')
    plt.show()
    plt.close()

pwr_dif = (fpga_pwr- pypymodel_pwr)/pypymodel_pwr

if f == '60khz':
    low = 467
    high = 472
elif f == '33khz':
    low = 256
    high = 261
elif f == '24khz':
    low = 185
    high = 191
elif f=="10khz":
    low = 77
    high = 82
elif f=="03khz":
    low = 21
    high = 27
elif f == '512hz':
    low = 3
    high = 6

#Simimg_comp = (model_img[low:high] - sim_img[low:high])/model_img[low:high]
#Simre_comp = (model_re[low:high] - sim_re[low:high])/model_re[low:high]
#Simpwr_comp = (pypymodel_pwr[low:high] - sim_pwr[low:high])/pypymodel_pwr[low:high]
print(fpga_re[low:high])
FPGAimg_compare = (model_img[low:high] - fpga_img[low:high])/model_img[low:high]
FPGAre_compare =   (model_re[low:high] - fpga_re[low:high])/model_re[low:high]
FPGA_pypypwr_compare = (pypymodel_pwr[low:high] - fpga_pwr[low:high])/pypymodel_pwr[low:high]
#FPGA_pyFPGApwr_compare = (pyFPGAmodel_pwr[low:high] - fpga_pwr[low:high])/pyFPGAmodel_pwr[low:high]

print(f+" FPGA imaginary offset", max(abs(FPGAimg_compare)))
#print(f+" Simulation imaginary offset", Simimg_comp)

print(f+" FPGA real offset", max(abs(FPGAre_compare)))
#print(f+" Simulation real offset", Simre_comp)

print(f+' FPGA power offset from pypy:\n', FPGA_pypypwr_compare)
#print(f+' FPGA power offset from pyFPGA:\n', FPGA_pyFPGApwr_compare)
#print(f+' Simulation Power Offset', Simpwr_comp)

#accumulated, averaged, and rebinned results - compressed and uncompressed

fpga_file_res = FPGA_base+'_avg_hex.txt'
fpga_comp,fpga_uncomp = read_FPGA_input_lines(fpga_file_res,64,line_n=3,x=1,y=2,signed=False)
fpga_uncomp = np.array(fpga_uncomp,dtype=np.int64)
fpga_comp = np.array(fpga_comp,dtype=np.int32)

pypy_file_rebin = pypy_base+'_avg_hex.txt'
pypy_file_comp = pypy_base+'_cmprs_hex.txt'
pypy_uncomp =  np.array(read_FPGA_input(pypy_file_rebin,64,False,show_plots=False),dtype=np.int64)
pypy_comp =  np.array(read_FPGA_input(pypy_file_comp,16,False,show_plots=False),dtype=np.int32)

#pyFPGA_file_rebin = pyFPGA_base+'_avg_hex.txt'
#pyFPGA_file_comp = pyFPGA_base+'_cmprs_hex.txt'
#pyFPGA_uncomp =  np.array(read_FPGA_input(pyFPGA_file_rebin,64,False,show_plots=False),dtype=np.int64)
#pyFPGA_comp =  np.array(read_FPGA_input(pyFPGA_file_comp,16,False,show_plots=False),dtype=np.int32)
transmitter = False
if f == '60khz':
    center = 56
elif f == '33khz':
    center = 53
elif f == '24khz':
    center = 48
    transmitter = True
    t_bin = 61
    center = t_bin
elif f == '10khz':
    center= 37
elif f == '03khz':
    center = 19
elif f == '512hz':
    center = 2


pypy_uncomp_delta = pypy_uncomp[center] - fpga_uncomp[center]
#pyFPGA_uncomp_delta = pyFPGA_uncomp[center] - fpga_uncomp[center]

FPGA_pypy_uncomp_compare = (pypy_uncomp_delta)/pypy_uncomp[center]
#FPGA_pyFPGA_uncomp_compare = (pyFPGA_uncomp_delta)/pyFPGA_uncomp[center]

pypy_comp_delta = pypy_comp[center] - fpga_comp[center]
#pyFPGA_comp_delta = pyFPGA_comp[center] - fpga_comp[center]

FPGA_pypy_comp_compare = (pypy_comp_delta )/pypy_comp[center]
#FPGA_pyFPGA_comp_compare = (pyFPGA_comp_delta)/pyFPGA_comp[center]
print('\n')
print('pypy uncompressed comparison: ', FPGA_pypy_uncomp_compare,"\n uncompressed delta: ",pypy_uncomp_delta )
print('pypy compressed comparison: ', FPGA_pypy_comp_compare,"\n compressed delta: ",pypy_comp_delta)

#print('pyFPGA uncompressed comparison: ', FPGA_pyFPGA_uncomp_compare,"\n uncompressed delta: ",pyFPGA_uncomp_delta)
#print('pyFPGA compressed comparison: ', FPGA_pyFPGA_comp_compare,"\n compressed delta: ",pyFPGA_comp_delta)

print("done")