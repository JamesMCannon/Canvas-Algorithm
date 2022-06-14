from tkinter import E
import serial #import serial library
import time
import numpy as np
from saveas import save_FFT, save_power, save_spectra, save_xspectra, saveall

def response_check(ser,ack,dump_line = True):
    msg_len = len(ack)
    ack_read = False
    val = ''
    while ack_read == False:
        if (ser.in_waiting > 0):
            if dump_line:
                if ser.in_waiting>msg_len: #clear to end of serial line
                    dump = ser.read(ser.in_waiting-msg_len) 
            
            val = ser.read(msg_len)
            if val == ack:
                ack_read = True
    return 

def ser_write(ser, command, len_header = True):
    if len_header:
        msg_len = len(command)
        header = msg_len.to_bytes(1,'big')
        ser.write(header)
    ser.write(command)
    return

def read_header(ser):
    #define sycn bytes
    sync = b'\x35\x2E\xF8\x53'

    #Synchronize with expected packet
    response_check(ser,sync,dump_line=False)

    #extract header info
    alg_id = ser.read(1)
    test_mode = ser.read(1)
    payload_len = ser.read(2)
    length = int.from_bytes(payload_len,'big') +1 #'big' => most significant byte is at the beginning of the byte array
    mask = b'\x0f' 
    test_mode = bytes([test_mode[0] & mask[0]])

    return length,test_mode

def readFPGA(ser, num_read = 1, readcon = 'none', outpath = 'HW-output/default-file'):
    #define data modes
    tx_packet_gen = b'\x02'
    rotation = b'\x03'
    fft_result = b'\x04'
    power_calc = b'\x05'
    acc_power = b'\x06'   
    spec_result = b'\x07'
    if readcon == 'all': #skip everything if Read All is chosen
        print('Read All')
        words = 16500
        readAll(words,ser,outpath='HW-output/read_all')
        return
        
    length,test_mode = read_header(ser)

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
    for i in range(num_read):
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
            vals = readFFT(words,ser,outpath)
        elif test_mode == power_calc:
            name = outpath + '_spectra' 
            vals = readPwr(words,ser,name)
        elif test_mode == acc_power:
            name = outpath + '_acc' #doesn't exist in 14p0 
            vals = readPwr(words,ser,name)
        elif test_mode == spec_result:
            vals = readSpec(words,ser,outpath)
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
        vals[i][5] = adc1
    return vals


def readFFT(words,ser,outpath):
    vals = np.zeros((words,3))
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        rFFT = int.from_bytes(ser.read(4),'big',signed=True)
        iFFT = int.from_bytes(ser.read(4),'big',signed=True)

        vals[i][0] = cur_bin
        vals[i][1] = rFFT
        vals[i][2] = iFFT
    save_FFT(vals,outpath+'_FFT',out_type='both')
    return vals

def readPwr(words,ser,outpath): #both with power and accumulated power
    vals = np.zeros((words,2),dtype=np.uint64)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        pwr = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = pwr
    save_power(vals,outpath,out_type='both')
    return vals

def readSpec(words,ser,outpath):
    vals = np.zeros((words,3),dtype=np.uint64)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        v=ser.read(2)
        mask = b'\x0f\xff'#4 bits unused 
        comp_rst = andbytes(v,mask)
        comp_rst = int.from_bytes(comp_rst,'big')
        uncomp_rst = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = comp_rst
        vals[i][2] = uncomp_rst
    save_spectra(vals,outpath + '_avg',out_type='both')
    return vals

def readXSpec(words,ser,outpath):
    vals = np.zeros((words,3),dtype=np.int64)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        v=ser.read(2)
        mask = b'\x0f\xff'#4 bits unused 
        comp_rst = andbytes(v,mask)
        comp_rst = int.from_bytes(comp_rst,'big') #uses sign-magnitude not two's compliment
        if (comp_rst>2048): #2^11 is 2048 so we use this as our comparison 
            comp_rst = (comp_rst-2048) 
        uncomp_rst = int.from_bytes(ser.read(8),'big',signed=True) #uses two's compliment

        vals[i][0] = cur_bin
        vals[i][1] = comp_rst
        vals[i][2] = uncomp_rst
    save_xspectra(vals,outpath + '_avg',out_type='both')
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

def readAll(words,ser,outpath): #basic read function, reads in two-byte intervals
    s1 = b'\x35'
    s2= b'\x2E'
    s3 = b'\xF8'
    s4 = b'\x53'
    vals = np.zeros((words,6),dtype=np.uint16)
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
                
    saveall(vals,outpath + '_avg',out_type='both')
    return vals


def andbytes(abytes, bbytes):
    return bytes([a & b for a, b in zip(abytes[::-1], bbytes[::-1])][::-1])