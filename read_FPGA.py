import numpy as np
import matplotlib.pyplot as plt

# python functions to read FPGA input files (in hex)

def twos_complement(hexstr,bits):
    value = int(hexstr,16) # hex is base 16
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value

def read_FPGA_input(file, bits, show_plot):
    f = open(file, 'r')
    datalines = [line for line in f]
    fpga_in_data = [twos_complement(p,bits) for p in datalines]
    f.close()

    if show_plot:
        plt.plot(np.log10(fpga_in_data[:512]),'.')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(fpga_in_data))

    return fpga_in_data