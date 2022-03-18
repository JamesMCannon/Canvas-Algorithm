from multiprocessing.connection import wait
from os import linesep
import sys
sys.path.append('C:/Users/James/Documents/Canvas/Canvas-Algorithm/') #import functions from parent folder
import serial #import serial library
import time
import numpy as np
from numpy import random
from serialfcns import wait4byte
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

s1 = b'\x35'
s2= b'\x2E'
s3 = b'\xF8'
s4 = b'\x53'

#define pic packet headers
SetConfig = '\x01'
Data = '\x02'
ResetPIC = '\x03'
StartFPGA = '\x04'
StopFPGA = '\x05'

#define pic SetConfig payloads
Nominal = '\x00'
GSE_Loopback = '\x01'
TX_Packet_Gen = '\x02'
Algorithm_Testing = '\x03'

channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output=None)
print(len(channels0_td))

pic_ser = serial.Serial("COM3",115200)
FPGA_ser = serial.Serial("COM4",115200)

#reset devices
#pic_ser.write(bytes(ResetPIC , 'utf-8'))
#pic_ser.write(bytes(ResetFPGA , 'utf-8'))

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(Algorithm_Testing , 'utf-8'))
#other command/payload?
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('Recieved command: ')
print(val)     

#start
pic_ser.write(bytes(StartFPGA , 'utf-8'))
pic_ser.write(bytes(lf, 'utf-8'))

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('Recieved command: ')
print(val)

#When to send buffer data?

#Synchronize with expected packet

r1=wait4byte(FPGA_ser,s1,False)
print('First Sync byte recieved: ')
print(r1)
r2=wait4byte(FPGA_ser,s2,False)
print('Second Sync byte recieved: ')
print(r2)
r3=wait4byte(FPGA_ser,s3,False)
print('Third Sync byte recieved: ')
print(r3)
r4=wait4byte(FPGA_ser,s4,False)
print('Fourth Sync byte recieved: ')
print(r4)

#extract header info
header = FPGA_ser.read(2)
payload_len = FPGA_ser.read(2)
length = int.from_bytes(payload_len,'big') +1 #'big' => most significant byte is at the beginning of the byte array
word_length = int(length/4)
print('Words in current packet:',word_length) #check work length

#read in payload in 4-byte words
vals = np.zeros(word_length)
for i in range(word_length):
    v=FPGA_ser.read(4)
    vals[i] = int.from_bytes(v,'big')

#save data
#np.savetxt('TestFile.csv', vals, delimiter=',')
v=int(vals[0])
print('First Entry: ',v.to_bytes(4, byteorder='big')) #Let's look at the last datum


