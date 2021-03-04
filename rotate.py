import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_fft_debug, read_FPGA_input_lines

def rotateSCM(fname):
    x,y = read_FPGA_input_lines(fname, 16, 6, 0, 1)
    z,u = read_FPGA_input_lines(fname, 16, 6, 2, 3)
    v,w = read_FPGA_input_lines(fname, 16, 6, 4, 5)

    Rm = np.array([[1,0,0],[0,1,0],[0,0,1]])

    for i in range(len(x)-1):
        xyz = np.array([x[i],y[i],z[i]])
        uvw = np.matmul(xyz,Rm)
        print(xyz, uvw, u[i],v[i],w[i])
    
rotateSCM('FPGA/adc_in_rotate_out.txt')