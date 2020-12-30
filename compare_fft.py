import numpy as np 
import matplotlib.pyplot as plt 


def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value


file = 'output/channel0_time.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_t = [twos_complement(p,64) for p in datalines]

file = 'fbin_accum_pwr.txt'
f = open(file, 'r')
datalines = [line for line in f]
fp_t = [twos_complement(p,64) for p in datalines]

py_t = np.array(py_t[:3584])
newpt = []
for i in range(0,len(py_t),512):
    pp = py_t[i+2:i+332]
    newpt.extend(pp)

newpt = np.array(newpt)
fp_t = np.array(fp_t)
dd = newpt - fp_t
print(newpt)
print(fp_t)

print(dd)

"""
file = 'output/channel0_time.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_t = [twos_complement(p,64) for p in datalines]

file = 'average_input.txt'
f = open(file, 'r')
datalines = [line for line in f]
fp_t = [twos_complement(p,64) for p in datalines]

py_t = np.array(py_t[:3584])
newpt = []
for i in range(0,len(py_t),512):
    pp = py_t[i+2:i+332]
    newpt.extend(pp)

newpt = np.array(newpt)
fp_t - np.array(fp_t)
dd = newpt - fp_t

print(dd)

plt.plot(np.log10(dd[:330]),'.')
plt.show()

file = 'spec_pwr_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fpga_in_data = [twos_complement(p,64) for p in datalines]

file = 'output/channel0_spectra.txt'
f = open(file, 'r')
datalines = [line for line in f]
py_data = [twos_complement(p,64) for p in datalines]
py_data = py_data[:4608]

file = 'output/channel0_fft_imag_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
pfy_data = [twos_complement(p,32) for p in datalines]

file = 'output/channel0_fft_real_hex.txt'
f = open(file, 'r')
datalines = [line for line in f]
pfx_data = [twos_complement(p,32) for p in datalines]

file = 'fft_sqrx_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fx_data = [twos_complement(p,32) for p in datalines]

file = 'fft_sqry_output.txt'
f = open(file, 'r')
datalines = [line for line in f]
fy_data = [twos_complement(p,32) for p in datalines]

fs = 2**17
nFFT = 1024
center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]

#py_data = np.array(py_data)
#fpga_in_data = np.array(fpga_in_data)
#plt.plot((py_data - fpga_in_data) / fpga_in_data)
#plt.show()
#plt.close()

c_py = [complex(pfx_data[i], pfy_data[i]) for i in range(len(fx_data))]
c_f = [complex(fx_data[i], fy_data[i]) for i in range(len(fx_data))]

plt.semilogy(center_freqs,np.abs(c_py[:512]),'.', label='python')
plt.semilogy(center_freqs,np.abs(c_f[:512]),linewidth=0.5, label='fpga')
plt.legend()
plt.title('FFT output')
plt.show()
plt.close()

plt.semilogy(center_freqs,py_data[:512],'.', label='python')
plt.semilogy(center_freqs,fpga_in_data[:512],linewidth=0.5, label='fpga')
plt.legend()
plt.title('power output')
plt.show()
plt.close()



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