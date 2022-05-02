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

from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_fft_debug, read_FPGA_input_lines


MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

# some set up parameters
fs = 131072.                # sampling freq. in Hz
signal_freq0 = 60000         # signal freq. 1 in Hz
amp0 = 2**15-1                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
sample_len = 0.5             # seconds
nFFT = 1024                 # length of FFT
n_acc = 8                   # number of FFTs to accummulate

#misc PIC commands
ack = b'\x06\x0A'
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
Ingress_Write = '\x00'
Ingress_Read = '\x01'
Ch_0_Pkt_Gen = '\x02'
ADC_And_Rotation = '\x03'
FFT_Results = '\x04'
FFT_Power = '\x05'
Average_Power = '\x06'
Specta_Results = '\x07'
Power_RAM_port_A = '\x08'
Power_RAM_port_B = '\x09'
Real_RAM_port_A = '\x0A'
Real_RAM_port_B = '\x0B'
X_Spec_Real_Results = '\x0C'
Imaginary_RAM_port_A = '\x0D'
Imaginary_RAM_port_B = '\x0E'
X_Spec_Imaginary_Results = '\x0F'

#Generate input signal from file or aribitrarily
fromFile = True

if fromFile:
    inputs = 'Inputs/'
    f = "60kHz"
    file = inputs+'hi_amp_'+f+'.txt'  
    channels0_td = read_FPGA_input(file,signed=True,show_plots=False)
else:
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
num_samples = len(channels0_td)
print(num_samples)
#num_samples = 11
test = channels0_td[0:num_samples]

#test = [i for i in range(num_samples)]


pic_ser = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM5",512000)

#configure PIC
testmode = Specta_Results
pic_ser.write(b'\x03')
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(testmode, 'utf-8')) #Change this 
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack,is_ascii=False)
print('FPGA Configured')

#Set number of samples to be buffered
pic_ser.write(b'\x06')
pic_ser.write(bytes(SetLength, 'utf-8'))
to_Send = num_samples.to_bytes(4,'big',signed=False)
pic_ser.write(num_samples.to_bytes(4,'big',signed=False))
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack,is_ascii=False)
print('Data Length Set')
t0=time.perf_counter()
#buffer data
var = 0
for i in test:
    val = i.to_bytes(2,byteorder='big',signed=True)
    data_len = b'\x07'
    to_write = data_len + Data + val + delim + val + lf
    pic_ser.write(to_write)
    if var%1000 == 0:
        print('buffering ', var)
    var = var+1
    #val=wait4byte(pic_ser,ack,is_ascii=False)

#check for complete from PIC
ready = b'Ready.\n'
ack_read = False
val = ''
while ack_read == False:
    if (pic_ser.in_waiting > 0):
        if pic_ser.in_waiting>7:
            dump = pic_ser.read(pic_ser.in_waiting-7)
        else:
            v = pic_ser.read(pic_ser.in_waiting)
            val=v
        if val == ready:
            ack_read = True


t1 = time.perf_counter()
del_t = t1-t0
print('Data buffered after %f seconds', del_t)

#start
pic_ser.write(b'\x02')
pic_ser.write(bytes(StartFPGA , 'utf-8'))
pic_ser.write(lf)

#Wait for acknowledge
val=wait4byte(pic_ser,ack,is_ascii=False) #PIC not sending ACK
print('FPGA Started')

vals,bits = readFPGA(FPGA_ser,readcon="none")
#vals1,bits1 = readFPGA(FPGA_ser,readAllcon = True)


out_folder = 'HW-output'
FPGA_rev = "Rev13p5"

if testmode == ADC_And_Rotation:
    sample = vals[:,0]
    adc2r = vals[:,1]
    adc1r = vals[:,2]
    adc2 = vals[:,3]
    adc1 = vals[:,4]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_ADC_And_Rotation'+f

    save_output_txt(sample,out_path+'sample','both',bits)
    save_output_txt(adc2r,out_path+'adc2r','both',bits)
    save_output_txt(adc1r,out_path+'adc1r','both',bits)
    save_output_txt(adc2,out_path+'adc2','both',bits)
    save_output_txt(adc1,out_path+'adc1','both',bits)

elif testmode == FFT_Results:
    bin= vals[:,0]
    re = vals[:,1]
    im = vals[:,2]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT'+f

    save_output_txt(bin,out_path+'bin','both',bits)
    save_output_txt(im,out_path+'img','both',bits)
    save_output_txt(re,out_path+'real','both',bits)

elif testmode == FFT_Power:
    bin = vals[:,0]
    pwr = vals[:,1]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT_PWR'+f

    save_output_txt(bin,out_path+'bin','both',bits)
    save_output_txt(pwr,out_path+'pwr','both',bits)

elif testmode == Average_Power:
    bin = vals[:,0]
    pwr = vals[:,1]

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_AVG_PWR'+f

    save_output_txt(bin,out_path+'bin','both',bits)
    save_output_txt(pwr,out_path+'pwr','both',bits)
elif testmode == Specta_Results:
    bin = vals[:,0] 
    comp_rst= vals[:,1] 
    uncomp_rst = vals[:,2] 

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_Spectra_Result'+f

    save_output_txt(bin,out_path+'bin','both',bits)
    save_output_txt(comp_rst,out_path+'compressed_result','both',bits)
    save_output_txt(uncomp_rst,out_path+'uncompressed_result','both',bits)


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


