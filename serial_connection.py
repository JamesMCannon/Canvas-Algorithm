from os import linesep
import serial #import serial library
import time
import numpy as np
from numpy import random

MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

ser = serial.Serial("COM3",115200)

# wait for start command
send_data = False
val = ''
while send_data == False:
    v = ser.read()
    val += v.decode('ascii')
    if v.decode('ascii') == '\n':
        print(val)
        if val == 'Send data.\n':
            send_data = True
            print('send data message received')
        else:
            val = ''
#create 5 arrays of values to pass to PIC
r1 = random.randint(MIN_VALUE_OF_16_BIT_INT, MAX_VALUE_OF_16_BIT_INT, size=(5))
r2 = random.randint(MIN_VALUE_OF_16_BIT_INT, MAX_VALUE_OF_16_BIT_INT, size=(5))
print(r1,r2)

ack = '\x06'
complete = '\nReady.'
send_complete = False
r_iterate = 0

# 5 arrays 
while send_complete == False:
    ser.write(int(r1[r_iterate]).to_bytes(2, 'little',signed=True))
    print(int(r1[r_iterate]).to_bytes(2, 'little',signed=True))
    print(int(r1[r_iterate]))
    ser.write(bytes(',' , 'utf-8'))
    ser.write(int(r2[r_iterate]).to_bytes(2, 'little',signed=True))
    ser.write(bytes('\n' , 'utf-8'))

    ack_read = False
    val = ''
    while ack_read == False:
        v = ser.read()
        val += v.decode('ascii')
        if v.decode('ascii') == ack:
            ack_read = True
            print(val)
            print('sending next array')
        if val == complete:
            ack_read = True
            send_complete = True
            print(val)
    r_iterate = r_iterate+1
print('sent all 5 pairs')