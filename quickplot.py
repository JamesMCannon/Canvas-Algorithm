from cProfile import label
from pyexpat import model
from readFPGA import read_FPGA_fft, read_FPGA_input
import numpy as np
import matplotlib.pyplot as plt

file_path = "./Data_compare/"
f = "60khz"
fpga_rev = "FPGA-Rev13p5"

simulation_file = file_path+f+'_fft_fbin_pwr.txt'

pymodel_file_img = file_path+'python_'+f+'_fft_imag_hex.txt'
pymodel_file_re = file_path+'python_'+f+'_fft_real_hex.txt'

fpga_file_img =  file_path+fpga_rev+'_FFT'+f+'real_hex.txt' #For 13p5 (and earlier?) real=img and img=real
fpga_file_re = file_path+fpga_rev+'_FFT'+f+'img_hex.txt'

bins = np.arange(512)
freq = bins*128/1000
sim_re,sim_img =np.array(read_FPGA_fft(simulation_file,32),dtype=np.int64)

model_img = np.array(read_FPGA_input(pymodel_file_img,32,True,show_plots=False),dtype=np.int64)
fpga_img = np.array(read_FPGA_input(fpga_file_img,32,True,show_plots=False),dtype=np.int64)

model_re = np.array(read_FPGA_input(pymodel_file_re,32,True,show_plots=False),dtype=np.int64)
fpga_re = np.array(read_FPGA_input(fpga_file_re,32,True,show_plots=False),dtype=np.int64)

model_img = model_img[:512]
model_re = model_re[:512]
sim_img = sim_img[0:512]
sim_re = sim_re[:512]
#plt.style.use('dark_background')

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


model_pwr = model_re**2 + model_img**2
sim_pwr = sim_re**2 + sim_img**2

fpga_file_pwr = file_path+fpga_rev+'_FFTPWR'+f+'pwr_hex.txt'
fpga_pwr = np.array(read_FPGA_input(fpga_file_pwr,64,False,show_plots=False),dtype=np.int64)
fpga_pwr = fpga_pwr[:512]
#dif = model_img - fpga_img

plt.plot(freq,model_pwr[:512],'-',label='Python')
plt.plot(freq,fpga_pwr[:512],'-',label='FPGA')
plt.plot(freq,sim_pwr[:512],'-',label = 'Simulation')
plt.title('Signal Power')
plt.yscale('log')
plt.legend()
plt.xlabel('Frequency (kHz)')
plt.show()
plt.close()
pwr_dif = (fpga_pwr- model_pwr)/model_pwr

if f == '60khz':
    low = 467
    high = 472
elif f == '33khz':
    low = 256
    high = 261
elif f == '512hz':
    low = 3
    high = 6

Simimg_comp = (model_img[low:high] - sim_img[low:high])/model_img[low:high]
Simre_comp = (model_re[low:high] - sim_re[low:high])/model_re[low:high]
Simpwr_comp = (model_pwr[low:high] - sim_pwr[low:high])/model_pwr[low:high]

FPGAimg_comp = (model_img[low:high] - fpga_img[low:high])/model_img[low:high]
FPGAre_comp =   (model_re[low:high] - fpga_re[low:high])/model_re[low:high]
FPGApwr_comp = (model_pwr[low:high] - fpga_pwr[low:high])/model_pwr[low:high]

print(f+" Max Imaginary Offset; FPGA", max(abs(FPGAimg_comp)))
print(f+" Max Imaginary Offset; Simulation", max(abs(Simimg_comp)))

print(f+" Max Real Offset; FPGA", max(abs(FPGAre_comp)))
print(f+" Max Real Offset; Simulation", max(abs(Simre_comp)))

print(f+' Max Power Offset; FPGA', max(abs(FPGApwr_comp)))
print(f+' Max Power Offset; Simulation', max(abs(Simpwr_comp)))
print("done")