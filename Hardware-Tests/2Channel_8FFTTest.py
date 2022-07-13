from doctest import testmod
from encodings import utf_8
from multiprocessing.connection import wait
import os
import sys
from urllib import response
sys.path.append("..\Canvas-Algorithm") #import functions from parent directory
import serial #import serial library
import time
import numpy as np
from numpy import random
#custom function imports
from saveas import save_output_txt
from serialfcns import readFPGA, ser_write, response_check
from inputstimulus import test_signal
from readFPGA import read_FPGA_input

# some set up parameters - used in signal generation
fs = 131072.                # sampling freq. in Hz
signal_freq0 = 60000         # signal freq. 1 in Hz
signal_freq1 = 60000         # signal freq. 1 in Hz
amp0 = 2**15-1                # amplitudes (in ADC units)
amp1 = 2**15-1                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
shift1 = 0                  # phase shift in radians
sample_len = 0.5             # seconds
xspec_test = True

#misc PIC commands
ack = b'\x06\x0A'
lf = b'\x0A'
delim = b'\x2C'
complete = b'Ready.\n'
initiated = b'\nInitiating.\n'

#define pic packet headers
SetConfig = b'\x01'
Data = b'\x02'
ResetPIC = b'\x03' #Just this, need to wait ~2 seconds after sending command
StartFPGA = b'\x04'
StopFPGA = b'\x05'
SetLength = b'\x06' #takes payload of uint32

#define pic SetConfig payloads
Ingress_Write = b'\x00'
Ingress_Read = b'\x01'
Ch_0_Pkt_Gen = b'\x02'
ADC_And_Rotation = b'\x03'
FFT_Results = b'\x04'
FFT_Power = b'\x05'
Spec_to_X_Spec_IF = b'\x06'
Spectra_Results = b'\x07'
Power_RAM_port_A = b'\x08'
Power_RAM_port_B = b'\x09'
Real_RAM_port_A = b'\x0A'
Real_RAM_port_B = b'\x0B'
X_Spec_Real_Results = b'\x0C'
Imaginary_RAM_port_A = b'\x0D'
Imaginary_RAM_port_B = b'\x0E'
X_Spec_Imaginary_Results = b'\x0F'

#GSE Commands to FPGA
Sync_Pat = b'\x1A\xCF\xFC\x1D'
Test_Enable = b'\x7E\x57\xBA\x11'
SW_Reset = b'\x7E\x57\xDE\xAD'
Config = b'\x7E\x57\xCF\x16'


#Generate input signal from file or aribitrarily
fromFile = True
num = 8

if fromFile:
    inputs = 'Inputs/'
    
    amp = "high-high_"
    phase = "5deg"
    f = "_03khz"
    file0 = inputs+amp+"0deg"+f+'.txt'
    file1 = inputs+amp+phase+f+'.txt' 
    channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
    channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)
else:
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
    channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
if len(channels0_td) > 20480:
    num_samples = 20480
else:
    num_samples = int(len(channels0_td))
print(num_samples)

test0 = channels0_td[0:num_samples]
test1 = channels1_td[0:num_samples]

#initialize serial ports
pic_ser = serial.Serial("COM5",115200)
FPGA_ser = serial.Serial("COM4",115200)


#main loop
spec_core = b'\x01'
iterate = 1
while iterate < 2:
    #set test mode
    if xspec_test:
        if iterate == 0:
            testmode = X_Spec_Real_Results
            mode = 'xspec_real'
        else: 
            #testmode = X_Spec_Imaginary_Results
            testmode = X_Spec_Real_Results #For testing, change this testmode
            readcon = 'none' #valid options are 'all' or 'none'. All dumps all data to a file, none proceeds with normal mode
            mode = 'xspec_imaginary'
    else:
        iterate+=1
        testmode = Spectra_Results
        mode = ''

    #reset FPGA
    ser_write(FPGA_ser,Sync_Pat+SW_Reset,False)
    print('FPGA Reset')

    #reset PIC
    time.sleep(0.5)
    ser_write(pic_ser,ResetPIC+lf,True)
    time.sleep(0.5)
    #response_check(pic_ser,ack)
    #print('Reset Received')
    response_check(pic_ser,initiated)
    print('PIC Reset')

    #Set number of samples to be buffered
    to_Send = num_samples.to_bytes(4,'big',signed=False)
    ser_write(pic_ser,SetLength+to_Send+lf)

    #Wait for acknowledge
    response_check(pic_ser,ack)
    print('Data Length Set')

    #buffer data
    t0=time.perf_counter()
    var = 0
    for i in range(len(test0)):
        val0 = test0[i].to_bytes(2,byteorder='big',signed=True)
        val1 = test1[i].to_bytes(2,byteorder='big',signed=True)
        ser_write(pic_ser,Data + val0 + delim + val1 + lf)
        if var%1000 == 0:
            print('buffering ', var)
        var = var+1
        #response_check(pic_ser,ack)

    #check for complete from PIC
    response_check(pic_ser,complete)

    t1 = time.perf_counter()
    del_t = t1-t0
    print('Data buffered after %f seconds', del_t)


    testmode = X_Spec_Imaginary_Results
    #configure FPGA
    ser_write(FPGA_ser,Sync_Pat+Config+spec_core+testmode,False)
    #time.sleep(2.5)

    #Start FPGA
    ser_write(FPGA_ser,Sync_Pat+Test_Enable,False)
    print('FPGA Started')
    #time.sleep(2.5)
 
    out_folder = 'HW-output'
    FPGA_rev = "Rev14p1_"

    vals = readFPGA(FPGA_ser,readcon=readcon,num_read=num,outpath=out_folder+'/FPGA-' + FPGA_rev + amp + f + mode)

    v=int(vals[0][0])
    print('First Entry: ',v) #Let's look at the first datum
    iterate+=1
print('Test Completed')


