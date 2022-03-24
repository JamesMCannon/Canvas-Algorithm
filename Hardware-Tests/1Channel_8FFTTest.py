from encodings import utf_8
from multiprocessing.connection import wait
import os
import sys
sys.path.append("..\Canvas-Algorithm") #import functions from parent directory
import serial #import serial library
import time
import numpy as np
from numpy import random
from serialfcns import wait4byte, readFPGA
from inputstimulus import test_signal


MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

# some set up parameters
fs = 131072.                # sampling freq. in Hz
signal_freq0 = 35e3         # signal freq. 1 in Hz
amp0 = 2**15                # amplitudes (in ADC units)
amp1 = 2**15                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
sample_len = 0.5             # seconds
nFFT = 1024                 # length of FFT
n_acc = 8                   # number of FFTs to accummulate

#misc PIC commands
ack = '\x06'
lf = '\x0A'
delim = '\x2C'
complete = '\nReady.'

#define pic packet headers
SetConfig = '\x01'
Data = '\x02'
ResetPIC = '\x03'
StartFPGA = '\x04'
StopFPGA = '\x05'
SetLength = '\x06' #takes payload of uint32

#define pic SetConfig payloads
Nominal = '\x00'
GSE_Loopback = '\x01'
TX_Packet_Gen = '\x02'
Rotation = '\x03'
FFT_result = '\x04'
Power_calc = '\x05'
Acc_power = '\x06'
Spectra_result = '\x07'

channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
num_samples = len(channels0_td)
print(num_samples)

#num_samples = 5
test = channels0_td[0:num_samples]

pic_ser = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM3",115200)

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(Spectra_result , 'utf-8'))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('FPGA Configured')

#Set number of samples to be buffered
pic_ser.write(bytes(SetLength, 'utf-8'))
pic_ser.write(num_samples.to_bytes(4,'big',signed=False))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('Data Length Set')

#buffer data
for i in test:
    pic_ser.write(bytes(Data, 'utf_8'))
    val = i.to_bytes(2,byteorder='big',signed=True)
    pic_ser.write(val) #buffer ADC1
    pic_ser.write(bytes(delim, 'utf-8'))
    pic_ser.write(val) #buffer ADC2
    pic_ser.write(bytes(lf, 'utf-8'))

    val=wait4byte(pic_ser,ack)

#check for complete from PIC
'''
send_complete = False

while send_complete == False:
    v = pic_ser.read()
    val += v.decode('ascii')
    if val == complete:
        send_complete = True
        print(val)
'''
print('Data buffered')

#start
pic_ser.write(bytes(StartFPGA , 'utf-8'))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('FPGA Started')

vals = readFPGA(FPGA_ser)

#save data
np.savetxt('TestFile.csv', vals, delimiter=',')
v=int(vals[0][1])
print('First Entry: ',v) #Let's look at the first datum


