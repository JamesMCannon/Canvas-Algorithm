from readFPGA import read_FPGA_fft_debug, read_FPGA_input
import numpy as np
import matplotlib.pyplot as plt

pymodel_file_img = 'channel0_fft_imag_hex.txt'
fpga_file_img =  'FFT_Result14000img_hex.txt'

pymodel_file_re = 'channel0_fft_real_hex.txt'
fpga_file_re =  'FFT_Result14000real_hex.txt'
model_img = np.array(read_FPGA_input(pymodel_file_img,32,True,show_plots=False),dtype=np.int64)
fpga_img = np.array(read_FPGA_input(fpga_file_img,32,True,show_plots=False),dtype=np.int64)
model_re = np.array(read_FPGA_input(pymodel_file_re,32,True,show_plots=False),dtype=np.int64)
fpga_re = np.array(read_FPGA_input(fpga_file_re,32,True,show_plots=False),dtype=np.int64)
model_img = model_img[512:1024]
model_re = model_re[512:1024]
model_pwr = model_img**2 + model_re**2
fpga_pwr = fpga_re**2 + fpga_img**2
fpga_pwr = fpga_pwr[512:1024]
#dif = model_img - fpga_img
freq = np.arange(512)
freq = freq*128
plt.plot(freq,model_pwr[:512],'-',color = 'blue')
plt.title('model')
plt.plot(freq,fpga_pwr[:512],'-',color = 'red')
plt.yscale('log')
plt.show()
plt.title('model')
pwr_dif = (fpga_pwr- model_pwr)/model_pwr
print("done")