import numpy as np
import matplotlib.pyplot as plt

# python functions to read FPGA input files (in hex)

# ---------------------------- 2's comp ---------------------------------------------
def twos_complement(hexstr,b):
    value = int(hexstr,16) # hex is base 16
    if value & (1 << (b-1)):
        value -= 1 << b
    return value
# ------------------------------------------------------------------------------------

# ---------------------------- read FPGA input ---------------------------------------
def read_FPGA_input(file, b, signed=True, show_plots=False):
    f = open(file, 'r')
    datalines = [line for line in f]
    if signed:
        fpga_in_data = [twos_complement(p,b) for p in datalines]
    else:
        fpga_in_data = [int(p,16) for p in datalines]

    f.close()

    if show_plots:
        plt.plot(fpga_in_data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(fpga_in_data))

    return fpga_in_data
# ------------------------------------------------------------------------------------

# ---------------------------- read INT input ---------------------------------------
def read_INT_input(file, show_plots=False):
    f = open(file, 'r')
    data = [int(line.strip('\n')) for line in f]
    f.close()

    if show_plots:
        plt.plot(data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(data))

    return data
# ------------------------------------------------------------------------------------

# ---------------------------- quick compare ---------------------------------------
def quick_compare(py_array, fp_array, vals, show_plots=False):
    py_array = np.array(py_array)
    fp_array = np.array(fp_array)
    
    diff = (py_array - fp_array)

    if show_plots:
        plt.plot(diff[:vals],'.')
        plt.title('difference in arrays - ' + str(vals) + ' vals')
        plt.show()
        plt.close()

    return diff
# ------------------------------------------------------------------------------------

def flatten(mylist):
    flat_list = [item for sublist in mylist for item in sublist]
    return flat_list