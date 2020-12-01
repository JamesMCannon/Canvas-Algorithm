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
file = 'channel1_fft_real_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]

py_x_data = [twos_complement(p,32) for p in datalines[:2650]]

file = 'channel1_fft_imag_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_y_data = [twos_complement(p,32) for p in datalines[:2560]]

cs_f_py = [complex(c_r, c_i) for c_r, c_i in zip(py_x_data, py_y_data)]
diffx = []
diffy = []
# let's shift to remove 0s
for i in range(len(cs_f_py)):
    thedx = (np.abs(py_x_data[i])) - (np.abs(fpga_x_data[i]))
    thedy = (np.abs(py_y_data[i])) - (np.abs(fpga_y_data[i]))
    if fpga_x_data[i] != 0:
        thdx = thedx / (np.abs(fpga_x_data[i]))
    else:
        thdx = 0
    if fpga_y_data[i] != 0:
        thdy = thedy / (np.abs(fpga_y_data[i]))
    else:
        thdy = 0
    diffx.append(thdx)
    diffy.append(thdy)

plt.plot(diffx,'.',label='x')
plt.plot(diffy,'.',label='y')
plt.legend()
plt.show()
plt.close()
print(max(diffx))
print(max(diffy))
indx = diffx.index(max(diffx))
indy = diffy.index(max(diffy))
print(indx)
print(indy)

print(py_x_data[indx], fpga_x_data[indx])
print(py_y_data[indy], fpga_y_data[indy])

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