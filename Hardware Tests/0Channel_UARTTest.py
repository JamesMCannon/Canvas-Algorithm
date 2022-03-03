from os import linesep
import serial #import serial library
import time
import numpy as np
from numpy import random

MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

ack = '\x06'
lf = '\x0A'

pic_ser = serial.Serial("COM3",115200)
FPGA_ser = serial.Serial("")

#define pic packet headers
SetConfig = '\x01'
Data = '\x02'
ResetPIC = '\x03'
ResetFPGA = '\x04'
StartFPGA = '\x05'
StopFPGA = '\x06'

#define pic SetConfig payloads
Idle = '\x00'
DummyPacket = '\x01'
TestMode1 = '\x02'
TestMode2 = '\x03'
TestMode3 = '\x04'
TestMode4 = '\x05'

#reset devices
#pic_ser.write(bytes(ResetPIC , 'utf-8'))
#pic_ser.write(bytes(ResetFPGA , 'utf-8'))

#configure PIC
pic_ser.write(bytes(SetConfig , 'utf-8'))
pic_ser.write(bytes(DummyPacket , 'utf-8'))

#TODO: wait for acknowledge, PIC will echo command sent

ack_read = False
val = ''
while ack_read == False:
    v = pic_ser.read()
    val += v.decode('ascii')
    if v.decode('ascii') == ack:
        ack_read = True
        print('Recieved command: ')
        print(val)
        

#start
pic_ser.write(bytes(StartFPGA , 'utf-8'))

#TODO: wait for acknowledge, PIC will echo command sent

length = 8 #length of packet in bytes

#read in packet
vals = np.zeros(length,1)
for i in length:
    v=pic_ser.read()
    vals[i] = int(v)

#save data
np.savetxt('TestFile.csv', vals, delimiter=',')
print(vals) #Let's look at the data


