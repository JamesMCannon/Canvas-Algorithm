import numpy as np 
import matplotlib.pyplot as plt 

def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value

# --------- ------- ------- FPGA FFT --------- ------- -------
file = 'fft_sqrx_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fpga_x_data = [twos_complement(p,32) for p in datalines]

file = 'fft_sqry_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fpga_y_data = [twos_complement(p,32) for p in datalines]

cs_f_fpga = [complex(c_r, c_i) for c_r, c_i in zip(fpga_x_data, fpga_y_data)]

# --------- ------- ------- python FFT --------- ------- -------
file = 'channel0_fft_real_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_x_data = [twos_complement(p,32) for p in datalines]

file = 'channel0_fft_imag_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_y_data = [twos_complement(p,32) for p in datalines]

cs_f_py = [complex(c_r, c_i) for c_r, c_i in zip(py_x_data, py_y_data)]

# ------ COMPARE ------
diffx = []
diffy = []

for i in range(len(fpga_x_data)):

    #if np.sign(py_y_data[i]) == np.sign(fpga_y_data[i]):
    dx = np.abs(py_x_data[i]) - np.abs(fpga_x_data[i])
    dy = np.abs(py_y_data[i]) - np.abs(fpga_y_data[i])

    if fpga_x_data[i] == 0 or fpga_y_data[i] == 0:
        diffx.append(dx / (1+np.abs(fpga_x_data[i])))
        diffy.append(dy / (1+np.abs(fpga_y_data[i])))
    else:
        diffx.append(dx / np.abs(fpga_x_data[i]))
        diffy.append(dy / np.abs(fpga_y_data[i]))

fs = 131072. 
nFFT = 1024

#plt.plot(diffx,'.',label='x')
#plt.plot(diffy,'.',label='y')
#plt.legend()
#plt.show()
#plt.close()
center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]
plt.semilogy(center_freqs,np.abs(cs_f_py[:512]))
plt.semilogy(center_freqs,np.abs(cs_f_fpga[:512]))
plt.show()

"""
the_x_diff = [(np.abs(fi) - np.abs(pi))/np.abs(fi) for fi, pi in zip(fpga_x_data, py_x_data)]
the_y_diff = [(np.abs(fi) - np.abs(pi))/np.abs(fi) for fi, pi in zip(fpga_y_data, py_y_data)]
#print(max(the_x_diff))
#print(the_y_diff)

#for i in range(len(cs_f_fpga)):
#    print(cs_f_py[i], cs_f_fpga[i])
# plot it 
fs = 131072
nFFT = 1024

#for i in range(len(cs_f_fpga)):
#    print(np.sign(np.imag(cs_f_fpga[i]))==np.sign(np.imag(cs_f_py[i])))

center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]
#plt.semilogy(center_freqs[1:nFFT//2], np.abs(cs_f_py[1:nFFT//2]), '-b', label='PYTHON')
#plt.semilogy(center_freqs[1:nFFT//2], np.abs(cs_f_fpga[1:nFFT//2]), '-r', label='FPGA')
#plt.title('FFT')
#plt.legend()
#plt.show()
#plt.close()

for i in range(0,len(cs_f_py), 512):
    plt.plot(center_freqs[:nFFT//2], (np.abs(np.real(cs_f_py[i:i+512])) - np.abs(np.real(cs_f_fpga[i:i+512])))/np.abs(np.real(cs_f_fpga[i:i+512])), '.', label='sqrx')
    plt.plot(center_freqs[:nFFT//2], (np.abs(np.imag(cs_f_py[i:i+512])) - np.abs(np.imag(cs_f_fpga[i:i+512])))/np.abs(np.imag(cs_f_fpga[i:i+512])), '.', label='sqry')
    plt.title('difference in output')
    plt.legend()
    #plt.show()
    plt.close()


for i in range(0,len(cs_f_py), 512):
    plt.plot(center_freqs[:nFFT//2], np.log10(np.abs(cs_f_py[i:i+512])), label='python')
    plt.plot(center_freqs[:nFFT//2], np.log10(np.abs(cs_f_fpga[i:i+512])), label='FPGA')
    plt.legend()

    #plt.show()
    plt.close()
"""