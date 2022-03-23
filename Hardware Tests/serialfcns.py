
import serial #import serial library
import time
import numpy as np

def wait4byte(ser,ack,is_ascii=True,timeout=10):
    
    ack_read = False
    val = ''
    #start = time.perf_counter()
    while ack_read == False:
        v = ser.read()
        if is_ascii:
            val = v.decode('ascii')
        else:
            val=v
        if val == ack:
            ack_read = True
        else:
            #curr_time = time.perf_counter
            #print(curr_time)
            #t_diff = curr_time-start
            t_diff = 5
            if t_diff>timeout:
                raise Exception("Timeout Error")
    return val

def readFPGA(ser):
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

    r1=wait4byte(ser,s1,False)
    r2=wait4byte(ser,s2,False)
    r3=wait4byte(ser,s3,False)
    r4=wait4byte(ser,s4,False)

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
        bins = False #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == rotation:
        print("rotation")
        word_length = 12 #bytes
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == fft_result:
        print("FFT Result")
        word_length = 12 #bytes
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == power_calc:
        print("Power Calculation")
        word_length = 12 #bytes
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == acc_power:
        print("Accumulated Power")
        word_length = 12 #bytes
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    elif test_mode == spec_result:
        print("Spectral Result")
        word_length = 12 #bytes
        bins = True #do the first 2 bytes of payload word denote sample/bin #?
    else:
        raise Exception("Unexpected Test Mode")

    words = length/word_length

    #read in payload 
    if test_mode==tx_packet_gen:
        raise Exception("Packet Gen not yet supported")
    elif test_mode == rotation:
        raise Exception("Rotation Data Product not yet supported")
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

    return vals


def readFFT(words,ser):
    vals = np.zeros(words,3)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        iFFT = int.from_bytes(ser.read(4),'big',signed=True)
        rFFT = int.from_bytes(ser.read(4),'big',signed=True)

        vals[i][0] = cur_bin
        vals[i][1] = iFFT
        vals[i][2] = rFFT
    return vals

def readPwr(words,ser): #both with power and accumulated power
    vals = np.zeros(words,2)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        unused = ser.read(2)
        pwr = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = pwr
    return vals


def readSpec(words,ser):
    vals = np.zeros(words,3)
    for i in range(words):
        cur_bin = int.from_bytes(ser.read(2),'big')
        v=ser.read(4)
        mask = b'\x0f\xff'
        comp_rst = andbytes(v,mask)
        uncomp_rst = int.from_bytes(ser.read(8),'big')

        vals[i][0] = cur_bin
        vals[i][1] = comp_rst
        vals[i][2] = uncomp_rst
    return vals


def andbytes(abytes, bbytes):
    return bytes([a & b for a, b in zip(abytes[::-1], bbytes[::-1])][::-1])