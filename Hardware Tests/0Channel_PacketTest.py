from multiprocessing.connection import wait
from os import linesep
import serial #import serial library
import time
import numpy as np
from numpy import random
from serialfcns import wait_byte

MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

ack = '\x06'
lf = '\x0A'

s1 = '\x1A'
s2= '\xCF'
s3 = '\xFC'
s4 = '\x1D'

pic_ser = serial.Serial("COM3",115200)
FPGA_ser = serial.Serial("")

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

#reset devices
#pic_ser.write(bytes(ResetPIC , 'utf-8'))
#pic_ser.write(bytes(ResetFPGA , 'utf-8'))

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(TX_Packet_Gen , 'utf-8'))

#Wait for acknowledge
val=wait_byte(pic_ser,ack)
print('Recieved command: ')
print(val)     

#start
pic_ser.write(bytes(StartFPGA , 'utf-8'))

#Wait for acknowledge
val=wait_byte(pic_ser,ack)
print('Recieved command: ')
print(val)

#Synchronize with expected packet
val = ''
val+=wait_byte(FPGA_ser,s1)
print('First Sync byte recieved: ')
print(val)
val+=wait_byte(FPGA_ser,s2)
print('Second Sync byte recieved: ')
print(val)
val+=wait_byte(FPGA_ser,s3)
print('Third Sync byte recieved: ')
print(val)
val+=wait_byte(FPGA_ser,s4)
print('Fourth Sync byte recieved: ')
print(val)

#extract header info
header = FPGA_ser.read(2)
print(header)#check header
payload_len = FPGA_ser.read(2)
length = int.from_bytes(payload_len,'big') -1 #'big' => most significant byte is at the beginning of the byte array
print(length) #check data length
word_length = length/4
print(word_length) #check work length

#read in payload
vals = np.zeros(word_length,1)
for i in word_length:
    v=FPGA_ser.read(4)
    vals[i] = int.from_bytes(v,'big')

#save data
np.savetxt('TestFile.csv', vals, delimiter=',')
print(v) #Let's look at the last datum


