from os import linesep
import serial #import serial library
import time
import random
import numpy as np

MAX_VALUE_OF_16_BIT_INT = 2 ** (16 - 1) - 1 # max for two's complement integer
MIN_VALUE_OF_16_BIT_INT = -1 * (2 ** (16 - 1)) # most negative for two's complement integer 

# This takes in two 16-bit integers and returns 
# two characters for each integer to represent them in ascii
# Example number: -76
# -76 in binary 2's comp: 1111_1111 1011_0100
# First, split the number into its first and second bytes
# Then, get the ascii character code for those numbers
def encodeNumbersIntoChars(one, two):
    if one > MAX_VALUE_OF_16_BIT_INT or \
       one < MIN_VALUE_OF_16_BIT_INT or \
       two > MAX_VALUE_OF_16_BIT_INT or \
       two < MIN_VALUE_OF_16_BIT_INT:
       raise Exception(f'The numbers must be between {MIN_VALUE_OF_16_BIT_INT} and {MAX_VALUE_OF_16_BIT_INT}')

    # We need to split each 16-bit int into two bytes
    # Ex: 0x54BC >> 8 = 0x0054; 0x0054 & 0xff = 0x54
    oneFirstByte = one >> 8 & 0xff
    # Ex: 0x54BC & 0xff = 0xBC
    oneSecondByte = one & 0xff

    twoFirstByte = two >> 8 & 0xff
    twoSecondByte = two & 0xff
           
    oneFirstChar = chr(oneFirstByte)
    oneSecondChar = chr(oneSecondByte)

    twoFirstChar = chr(twoFirstByte)
    twoSecondChar = chr(twoSecondByte)

    return [oneFirstChar, oneSecondChar, twoFirstChar, twoSecondChar]


if __name__ == "__main__":
    print('Testing encode function with integers 16, -128')
    print(encodeNumbersIntoChars(178, 66))


ser = serial.Serial("/dev/tty.usbserial-FTBIE0XW",115200)

# ---------------------------- 2's comp ---------------------------------------------
def twos_complement(value,b):
    #value = int(val,16) # hex is base 16
    if value & (1 << (b-1)):
        value -= 1 << b
    return value
# ------------------------------------------------------------------------------------

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

# 5 arrays 
for i in range(5):
    r1 = random.randint(MIN_VALUE_OF_16_BIT_INT, MAX_VALUE_OF_16_BIT_INT)
    r2 = random.randint(MIN_VALUE_OF_16_BIT_INT, MAX_VALUE_OF_16_BIT_INT)
    
    print(r1,r2)

    # Does this work? May not even have to encode the whole thing here tbh; 
    # looks like this to_bytes function could do it?
    ser.write((r1).to_bytes(2, 'little',signed=True))
    ser.write(bytes(',' , 'utf-8'))
    ser.write((r2).to_bytes(2, 'little',signed=True))
    ser.write(bytes('\n' , 'utf-8'))

    ack_read = False
    while ack_read == False:
        ack = '\x06'
        val = ser.read()
        if val.decode('ascii') == ack:
            ack_read = True
            print(val)
            print('sending next array')

print('send all 5 pairs')