from encodings import utf_8
from multiprocessing.connection import wait
from os import linesep
import sys
sys.path.append('C:/Users/James/Documents/Canvas/Canvas-Algorithm/') #import functions from parent folder
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
ack = '\x06'
lf = '\x0A'

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
Algorithm_Testing = '\x03'

channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
print(len(channels0_td))

pic_ser = serial.Serial("COM3",115200)
FPGA_ser = serial.Serial("COM4",115200)

#reset devices
#pic_ser.write(bytes(ResetPIC , 'utf-8'))
#pic_ser.write(bytes(ResetFPGA , 'utf-8'))

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(Algorithm_Testing , 'utf-8'))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('Recieved command: ')
print(val)     


#buffer data
for i in channels0_td:
    pic_ser.write(bytes(Data, 'utf_8'))
    val = channels0_td[i].to_bytes(2,byteorder='big',signed=True)
    pic_ser.write(val)
    pic_ser.write(',','utf-8')
    pic_ser.write(val)
    pic_ser.write(bytes(lf, 'utf-8'))

    val=wait4byte(pic_ser,ack)

print('Data buffered')

#start
pic_ser.write(bytes(StartFPGA , 'utf-8'))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('Recieved command: ')
print(val)
#Synchronize with expected packet

vals = readFPGA(ser)

#save data
np.savetxt('TestFile.csv', vals, delimiter=',')
v=int(vals[0][1])
print('First Entry: ',v) #Let's look at the first datum


