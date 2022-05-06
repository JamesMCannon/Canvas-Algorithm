
from tkinter import E
import serial #import serial library
import time
import numpy as np

def wait4byte(ser,ack,is_ascii=True,byte_size=2):
    ready = b'Ready.\n'
    ack_read = False
    val = ''
    while ack_read == False:
        if (ser.in_waiting > 0):
            v = ser.read(byte_size)
            if is_ascii:
                val = v.decode('ascii')
            else:
                val=v
            if val == ack or val == ready:
                ack_read = True
    return val

def readFPGA(ser, readcon = 'none'):
    #define data modes
    tx_packet_gen = b'\x02'
    rotation = b'\x03'
    fft_result = b'\x04'
    power_calc = b'\x05'
    acc_power = b'\x06'   
    spec_result = b'\x07'

    #define sycn bytes
    s1 = b'\x35'
    s2= b'\x2E'
    s3 = b'\xF8'
    s4 = b'\x53'

    #Synchronize with expected packet
    r1=wait4byte(ser,s1,False,1)
    r2=wait4byte(ser,s2,False,1)
    r3=wait4byte(ser,s3,False,1)
    r4=wait4byte(ser,s4,False,1)

    #extract header info
    alg_id = ser.read(1)
    test_mode = ser.read(1)
    payload_len = ser.read(2)
    length = int.from_bytes(payload_len,'big') +1 #'big' => most significant byte is at the beginning of the byte array
    mask = b'\x0f' 
    test_mode = bytes([test_mode[0] & mask[0]])

    if test_mode==tx_packet_gen:
        print("tx_packet_gen")
        word_length = 4 #bytes
        bits  = 'u-16'
        bins = False #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == rotation:
        print("rotation")
        word_length = 12 #bytes
        bits = 's-16'
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == fft_result:
        print("FFT Result")
        word_length = 12 #bytes
        bits = 's-32'
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == power_calc:
        print("Power Calculation")
        word_length = 12 #bytes
        bits = 'u-64'
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == acc_power:
        print("Accumulated Power")
        word_length = 12 #bytes
        bits = 'u-64'
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == spec_result:
        print("Spectral Result")
        word_length = 12 #bytes
        bits = 'u-64'
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif readcon == 'all':
        print("new test mode, read all")
        word_length = 12 #dummy, gets over written in 91
    elif readcon == '12':
        print("new test mode, reading 12 byte words")
        word_length = 12 #dummy, gets over written in 91
    else:
        print("Unexpected Test Mode - Forcing ReadAll")
        word_length = 12
        readcon = 'all'
        #raise Exception("Unexpected Test Mode")

    words = int(length/word_length)

    #read in payload 
    if readcon == 'all':
        words=16500
        vals = readAll(words,ser)
        bits = 'u-16'
    elif readcon == '12':
        words=16500
        vals = read12(words,ser)
        bits = 's-32'
    elif test_mode==tx_packet_gen:
        raise Exception("Packet Gen not yet supported")
    elif test_mode == rotation:
        vals = readRotate(words,ser)
    elif test_mode == fft_result:
        vals = readFFT(words,ser)
    elif test_mode == power_calc:
        vals = readPwr(words,ser)
    elif test_mode == acc_power:
        vals = readPwr(words,ser)
    elif test_mode == spec_result:
        vals = readSpec(words,ser)
    else:
        raise Exception("Unexpected Test Mode")

    return vals,bits

def readRotate(words,ser):
    vals = np.zeros((words,6))
    for i in range(words):
        adc3_r = int.from_bytes(ser.read(2), 'big')
        adc2_r = int.from_bytes(ser.read(2), 'big')
        adc1_r = int.from_bytes(ser.read(2),'big')
        adc3 = int.from_bytes(ser.read(2), 'big')
        adc2 = int.from_bytes(ser.read(2), 'big')
        adc1 = int.from_bytes(ser.read(2), 'big')

        vals[i][0] = adc3_r
        vals[i][1] = adc2_r
        vals[i][2] = adc1_r
        vals[i][3] = adc3
        vals[i][4] = adc2
        vals[i][5] = adc3
    return vals


def readFFT(words,ser):
    vals = np.zeros((words,3))
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        rFFT = int.from_bytes(ser.read(4),'big',signed=True)
        iFFT = int.from_bytes(ser.read(4),'big',signed=True)

        vals[i][0] = cur_bin
        vals[i][1] = rFFT
        vals[i][2] = iFFT
    return vals

def readPwr(words,ser): #both with power and accumulated power
    vals = np.zeros((words,2))
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        pwr = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = pwr
    return vals


def readSpec(words,ser):
    vals = np.zeros((words,3))
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        v=ser.read(4)
        mask = b'\x0f\xff'
        comp_rst = andbytes(v,mask)
        comp_rst = int.from_bytes(comp_rst,'big')
        uncomp_rst = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = comp_rst
        vals[i][2] = uncomp_rst
    return vals

def read12(words, ser):
    vals  = np.zeros((words,3),dtype=np.uint32)
    for i in range(words):
        if i%1000==0:
            print("reading vals ", i)
        v1 = ser.read(4)
        v2 = ser.read(4)
        v3 = ser.read(4)

        vals[i][0] = int.from_bytes(v1, 'big',signed=True)
        vals[i][1] = int.from_bytes(v2, 'big',signed=True)
        vals[i][2] = int.from_bytes(v3, 'big',signed=True)
    return vals

def readAll(words,ser): #basic read function, reads in two-byte intervals
    s1 = b'\x35'
    s2= b'\x2E'
    s3 = b'\xF8'
    s4 = b'\x53'
    vals = np.zeros((words,6))
    #vals = bytearray()
    for i in range(words):
            if i%1000==0:
                print("reading vals ", i)
            v0 = ser.read(2)
            vals[i][0] = int.from_bytes(v0,'big')
            if v0 == (s1+s2):
                v1 = ser.read(2)
                vals[i][1] = int.from_bytes(v1,'big')
                if v1 == (s3+s4):
                    low = 2
                    high = 4
                else:
                    low = 2
                    high = 6
            else:
                low = 1
                high = 6
            for j in range(low,high):
                v = ser.read(2)
                vals[i][j] = int.from_bytes(v,'big')
                
            '''
            v1 = ser.read(2)
            v2 = ser.read(2)
            v3 = ser.read(2)
            v4 = ser.read(2)
            v5 = ser.read(2)
            
            vals[i][0] = v0
            vals[i][1] = v1
            vals[i][2] = v2
            vals[i][3] = v3
            vals[i][4] = v4
            vals[i][5] = v5
            
            vals[i][0] = int.from_bytes(v0,'big')
            vals[i][1] = int.from_bytes(v1,'big')
            vals[i][2] = int.from_bytes(v2,'big')
            vals[i][3] = int.from_bytes(v3,'big')
            vals[i][4] = int.from_bytes(v4,'big')
            vals[i][5] = int.from_bytes(v5,'big')
            '''
    return vals


def andbytes(abytes, bbytes):
    return bytes([a & b for a, b in zip(abytes[::-1], bbytes[::-1])][::-1])