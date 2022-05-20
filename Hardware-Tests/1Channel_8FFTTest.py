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
amp0 = 2**15-1                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
sample_len = 0.5             # seconds

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
#Average_Power = b'\x06' #no longer supported
Specta_Results = b'\x07'
Power_RAM_port_A = b'\x08'
Power_RAM_port_B = b'\x09'
Real_RAM_port_A = b'\x0A'
Real_RAM_port_B = b'\x0B'
X_Spec_Real_Results = b'\x0C'
Imaginary_RAM_port_A = b'\x0D'
Imaginary_RAM_port_B = b'\x0E'
X_Spec_Imaginary_Results = b'\x0F'

#Generate input signal from file or aribitrarily
fromFile = True
testmode = Specta_Results
num = 1

if fromFile:
    inputs = 'Inputs/'
    f = "512Hz"
    amp = "hi_amp_"
    file = inputs+amp+f+'.txt'  
    channels0_td = read_FPGA_input(file,signed=True,show_plots=False)
else:
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
if len(channels0_td) > 20480:
    num_samples = 20480
else:
    num_samples = int(len(channels0_td))
print(num_samples)
#num_samples = 11
test = channels0_td[0:num_samples]
#test = [i for i in range(num_samples)]

#initialize serial ports
pic_ser = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM5",512000)

#reset PIC
time.sleep(0.5)
ser_write(pic_ser,ResetPIC+lf,True)
time.sleep(0.5)
#response_check(pic_ser,ack)
#print('Reset Received')
response_check(pic_ser,initiated)
print('PIC Reset')

#configure PIC
ser_write(pic_ser,SetConfig+testmode+lf)

#Wait for acknowledge
response_check(pic_ser,ack)
print('FPGA Configured')

#Set number of samples to be buffered
to_Send = num_samples.to_bytes(4,'big',signed=False)
ser_write(pic_ser,SetLength+to_Send+lf)

#Wait for acknowledge
response_check(pic_ser,ack)
print('Data Length Set')

#buffer data
t0=time.perf_counter()
var = 0
for i in test:
    val = i.to_bytes(2,byteorder='big',signed=True)
    ser_write(pic_ser,Data + val + delim + val + lf)
    if var%1000 == 0:
        print('buffering ', var)
    var = var+1
    #response_check(pic_ser,ack)

#check for complete from PIC
response_check(pic_ser,complete)

t1 = time.perf_counter()
del_t = t1-t0
print('Data buffered after %f seconds', del_t)

#start
ser_write(pic_ser,StartFPGA+lf)

#Wait for acknowledge
response_check(pic_ser,ack)
print('FPGA Started')

out_folder = 'HW-output'
FPGA_rev = "Rev14p1_"


vals,bits = readFPGA(FPGA_ser,readcon="none",num_read=num,outpath=out_folder+'/FPGA-' + FPGA_rev + amp + f)

if testmode == ADC_And_Rotation:
    adc3r = vals[:,0]
    adc2r = vals[:,1]
    adc1r = vals[:,2]
    adc3 = vals[:,3]
    adc2 = vals[:,4]
    adc1 = vals[:,5]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_ADC_And_Rotation'+f

    save_output_txt(adc3r,out_path+'adc3r','both',bits)
    save_output_txt(adc2r,out_path+'adc2r','both',bits)
    save_output_txt(adc1r,out_path+'adc1r','both',bits)
    save_output_txt(adc3,out_path+'adc3','both',bits)
    save_output_txt(adc2,out_path+'adc2','both',bits)
    save_output_txt(adc1,out_path+'adc1','both',bits)

elif testmode == FFT_Results:
    bin= vals[:,0]
    re = vals[:,1]
    im = vals[:,2]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT'+f


elif testmode == FFT_Power:
    bin = vals[:,0]
    pwr = vals[:,1]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT_PWR'+f

elif testmode == Specta_Results:
    bin = vals[:,0] 
    comp_rst= vals[:,1] 
    uncomp_rst = vals[:,2] 

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_Spectra_Result'+f



'''all1 = vals[:,0]
all2 = vals[:,1]
all3 = vals[:,2]
all4 = vals[:,3]
all5 = vals[:,4]
all6 = vals[:,5]'''

'''word1 = vals[:,0]
word2 = vals[:,1]
word3 = vals[:,2]'''


#save data


'''save_output_txt(all1,out_path+'1','both',bits)
save_output_txt(all2,out_path+'2','both',bits)
save_output_txt(all3,out_path+'3','both',bits)
save_output_txt(all4,out_path+'4','both',bits)
save_output_txt(all5,out_path+'5','both',bits)
save_output_txt(all6,out_path+'6','both',bits)
'''


'''out_path = out_folder+'/FPGA-Rev13p5_Specta_Results12_'+f
save_output_txt(word1,out_path+'Word_1','both',bits)
save_output_txt(word2,out_path+'Word_2','both',bits)
save_output_txt(word3,out_path+'Word_3','both',bits)
'''
v=int(vals[0][0])
print('First Entry: ',v) #Let's look at the first datum


