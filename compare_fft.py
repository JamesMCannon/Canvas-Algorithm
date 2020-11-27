import numpy as np 
import matplotlib.pyplot as plt 

def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value

# FPGA FFT
file = 'fft_sqrx_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fpga_x_data = [twos_complement(p,32) for p in datalines]

file = 'fft_sqry_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fpga_y_data = [twos_complement(p,32) for p in datalines]

cs_f_fpga = [complex(c_r, c_i) for c_r, c_i in zip(fpga_x_data, fpga_y_data)]


# python FFT
file = 'channel1_fft_real_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_x_data = [twos_complement(p,32) for p in datalines[:4608]]

file = 'channel1_fft_imag_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_y_data = [twos_complement(p,32) for p in datalines[:4608]]

cs_f_py = [complex(c_r, c_i) for c_r, c_i in zip(py_x_data, py_y_data)]

for i in range(len(cs_f_fpga)):
    print(cs_f_py[i], cs_f_fpga[i])
# plot it 
fs = 131072
nFFT = 1024

for i in range(len(cs_f_fpga)):
    print(np.sign(np.imag(cs_f_fpga[i]))==np.sign(np.imag(cs_f_py[i])))

center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]
#plt.semilogy(center_freqs[1:nFFT//2], np.abs(cs_f_py[1:nFFT//2]), '-b', label='PYTHON')
#plt.semilogy(center_freqs[1:nFFT//2], np.abs(cs_f_fpga[1:nFFT//2]), '-r', label='FPGA')
#plt.title('FFT')
#plt.legend()
#plt.show()
#plt.close()

#plt.plot(np.abs(np.real(cs_f_py)) - np.abs(np.real(cs_f_fpga)), '.', label='sqrx')
#plt.plot(np.abs(np.imag(cs_f_py)) - np.abs(np.imag(cs_f_fpga)), '.', label='sqry')
#plt.title('difference in output')
#plt.legend()
#plt.show()
#plt.close()

for i in range(0,len(cs_f_py), 512):
    plt.plot(center_freqs[:nFFT//2], np.log10(np.abs(cs_f_py[i:i+512])), label='python')
    plt.plot(center_freqs[:nFFT//2], np.log10(np.abs(cs_f_fpga[i:i+512])), label='FPGA')
    plt.legend()

    #plt.show()
    plt.close()
