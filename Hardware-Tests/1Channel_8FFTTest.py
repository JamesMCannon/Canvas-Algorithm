from encodings import utf_8
from multiprocessing.connection import wait
import os
import sys
sys.path.append("..\Canvas-Algorithm") #import functions from parent directory
import serial #import serial library
import time
import numpy as np
from numpy import random
from saveas import save_output_txt
from serialfcns import wait4byte, readFPGA
from inputstimulus import test_signal


MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

# some set up parameters
fs = 131072.                # sampling freq. in Hz
signal_freq0 = fs/4         # signal freq. 1 in Hz
amp0 = 2**15-1                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
sample_len = 0.5             # seconds
nFFT = 1024                 # length of FFT
n_acc = 8                   # number of FFTs to accummulate

#misc PIC commands
ack = b'\x06'
lf = b'\x0A'
delim = b'\x2C'
complete = '\nReady.'

#define pic packet headers
SetConfig = '\x01'
Data = b'\x02'
ResetPIC = '\x03' #Just this, need to wait ~2 seconds after sending command
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

channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output=None)
num_samples = len(channels0_td)
print(num_samples)
num_samples = 11
test = channels0_td[0:num_samples]

pic_ser = serial.Serial("COM4",460800)
FPGA_ser = serial.Serial("COM5",115200)

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(Rotation , 'utf-8'))
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack,is_ascii=False)
print('FPGA Configured')

#Set number of samples to be buffered
pic_ser.write(bytes(SetLength, 'utf-8'))
to_Send = num_samples.to_bytes(4,'big',signed=False)
pic_ser.write(num_samples.to_bytes(4,'big',signed=False))
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack,is_ascii=False)
print('Data Length Set')

#buffer data
var = 0
for i in test:
    val = i.to_bytes(2,byteorder='big',signed=True)
    to_write = Data + val + delim + val + lf
    pic_ser.write(to_write)
    #pic_ser.write(bytes(Data, 'utf_8'))
    #val = i.to_bytes(2,byteorder='big',signed=True)
    #pic_ser.write(val) #buffer ADC1
    #pic_ser.write(bytes(delim, 'utf-8'))
    #pic_ser.write(val) #buffer ADC2
    #pic_ser.write(bytes(lf, 'utf-8'))
    if var%1000 == 0:
        print('buffering ', var)
    var = var+1
    val=wait4byte(pic_ser,ack,is_ascii=False)

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
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack)
print('FPGA Started')

vals,bits = readFPGA(FPGA_ser,readAll=True)

#save data
out_folder = 'HW-output'
out_path = out_folder+'/channel'+'_cmprs'
save_output_txt(vals,out_path,'Both',bits)
v=int(vals[0][1])
print('First Entry: ',v) #Let's look at the first datum


