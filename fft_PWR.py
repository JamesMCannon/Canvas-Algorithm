import numpy as np
import matplotlib.pyplot as plt
from read_FPGA import read_FPGA_input

# function to take spectra power 
# input should be 1D length arrays 
def fft_spec_power(real_data, imag_data, show_plot):

    spec_pwr = np.zeros_like(real_data)

    for ind, (r,i) in enumerate(zip(real_data, imag_data)): 
        spec_pwr[ind] = r**2 + i**2

    if show_plot:
        plt.plot(np.log10(spec_pwr),'.')
        plt.title('spectra power')
        plt.show()
        plt.close()

    return spec_pwr
  
nFFT = 1024

# get the real input
in_reldata = read_FPGA_input('FPGA/fbin_fft_real.txt', 32, False)

# get the imaginary input
in_imgdata = read_FPGA_input('FPGA/fbin_fft_imgry.txt', 32, False)

all_spec_pwr = []
for n in range(0,len(in_reldata), nFFT//2):
    spec_pwr = fft_spec_power(in_reldata[n:n+512], in_imgdata[n:n+512], False)
    all_spec_pwr.append(spec_pwr)

print('spectra power array shape is: ', np.shape(all_spec_pwr))

# compare w FPGA output
fpga_pwr = read_FPGA_input('FPGA/fbin_fft_pwr.txt', 64, False)

for n in range(61):
    plt.plot(np.log10(all_spec_pwr[n]))
    plt.plot(np.log10(fpga_pwr[n*512:512*(n+1)]),'.')
    plt.title(str(n))
    #plt.show()
    plt.close()

    print(all_spec_pwr[n] - fpga_pwr[n*512:512*(n+1)])


