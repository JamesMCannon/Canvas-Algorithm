import numpy as np
import matplotlib.pyplot as plt
import os.path
# ---------------------------- save output text files ------------------------------

# options for bits are s-16, u-16, s-32, u-64
# easier to make these funcs APPEND so make sure the files are CLEARED before running

def save_output_txt(out_array, out_path, out_type, bits): 
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'
        with open(out_name, 'a') as output:
            for x in out_array:
                if bits == 's-16':
                    output.write(format(np.int16(x) & 0xffff, '04X') + '\n')
                if bits == 'u-16':
                    output.write(format(np.uint16(x) & 0xffff, '04X') + '\n')
                if bits == 's-32':
                    output.write(format(np.int32(x) & 0xffffffff, '08X') + '\n')
                if bits == 'u-64':
                    output.write(format(np.uint64(x) & 0xffffffffffffffff, '016X') + '\n')
                if bits == 's-64':
                    output.write(format(np.uint64(x) & 0xffffffffffffffff, '016X')+ '\n')    
    if out_type =='int' or out_type =='both':
        out_name = out_path+'_int.txt'
        with open(out_name, 'a') as output:
            for x in out_array:
                if bits == 's-16':
                    output.write(str(np.int16(x)) + '\n')
                if bits == 'u-16':
                    output.write(str(np.uint16(x)) + '\n')
                if bits == 's-32':
                    output.write(str(np.int32(x)) + '\n')
                if bits == 'u-64':
                    output.write(str(np.uint64(x)) + '\n')
                if bits == 's-64':
                    output.write(str(np.int64(x))+ '\n')
    return
# ------------------------------------------------------------------------------------

def save_rotate(out_array, out_path, out_type):
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('R3' + '\t') #Rotated ADC3
                output.write('R2' + '\t') #Rotated ADC2
                output.write('R1' + '\t') #Rotated ADC1
                output.write('A3' + '\t') #ADC3
                output.write('A2' + '\t') #ADC2
                output.write('A1' + '\n') #ADC1 
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.int16(x[0]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int16(x[1]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int16(x[2]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int16(x[3]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int16(x[4]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int16(x[5]) & 0xffff, '04X') + '\n') #bin number

    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('R3' + '\t') #Rotated ADC3
                output.write('R2' + '\t') #Rotated ADC2
                output.write('R1' + '\t') #Rotated ADC1
                output.write('A3' + '\t') #ADC3
                output.write('A2' + '\t') #ADC2
                output.write('A1' + '\n') #ADC1  

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.int16(x[0])) + '\t') #bin number
                output.write(str(np.int16(x[1])) + '\t') #bin number
                output.write(str(np.int16(x[2])) + '\t') #bin number
                output.write(str(np.int16(x[3])) + '\t') #bin number
                output.write(str(np.int16(x[4])) + '\t') #bin number
                output.write(str(np.int16(x[5])) + '\n') #bin number
    return

def save_FFT(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t') #bin number
                output.write('FFTr' + '\t\t') #FFTr 
                output.write('FFTi' + '\n') #FFTi 
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.int32(x[1]) & 0xffffffff, '08X') + '\t') #FFTr 
                output.write(format(np.int32(x[2]) & 0xffffffff, '08X') + '\n') #FFTi 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t') #bin number
                output.write('FFTr' + '\t') #FFTr 
                output.write('FFTi' + '\n') #FFTi 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t') #bin number
                output.write(str(np.int32(x[1])) + '\t') #FFTr 
                output.write(str(np.int32(x[2])) + '\n') #FFTi 
    return

def save_power(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t') #bin number
                output.write('Power' + '\n') #FFT Power 
   
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.uint64(x[1]) & 0xffffffffffffffff, '016X') + '\n') #Power 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t') #bin number
                output.write('Power' + '\n') #FFT Power 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t') #bin number
                output.write(str(np.uint64(x[1])) + '\n') #FFT power 
    return


def save_spectra(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('SBin' + '\t') #bin number
                output.write('Comp' + '\t\t') #Compressed result 
                output.write('Uncomp' + '\n') #Uncompressed result
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t') #Bin number
                output.write(format(np.uint32(x[1]) & 0xffffffff, '08X') + '\t') #Compressed result
                output.write(format(np.uint64(x[2]) & 0xffffffffffffffff, '016X') + '\n') #Uncompressed result 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('SBin' + '\t') #Bin number
                output.write('Comp' + '\t') #Compressed result 
                output.write('Uncomp' + '\n') #Uncompressed result 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t') #Bin number
                output.write(str(np.uint32(x[1])) + '\t') #Compressed result
                output.write(str(np.uint64(x[2])) + '\n') #Uncompressed result 
    return

def save_IF(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t')         #bin number
                output.write('Last' + '\t')     #Last FFT Indicator
                output.write('End' + '\t')      #End of FFT Indicator
                output.write('Start' + '\t')    #Start of FFT Indicator
                output.write('FFTr' + '\t\t')       #FFTr 
                output.write('FFTi' + '\n')         #FFTi 
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.uint16(x[1]) & 0xffff, '01X') + '\t') #Last FFT Indicator
                output.write(format(np.uint16(x[2]) & 0xffff, '01X') + '\t') #End of FFT Indicator
                output.write(format(np.uint16(x[3]) & 0xffff, '01X') + '\t') #Start of FFT Indicator
                output.write(format(np.int32(x[4]) & 0xffffffff, '08X') + '\t') #FFTr 
                output.write(format(np.int32(x[5]) & 0xffffffff, '08X') + '\n') #FFTi 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('FBin' + '\t')         #bin number
                output.write('Last' + '\t')     #Last FFT Indicator
                output.write('End' + '\t')      #End of FFT Indicator
                output.write('Start' + '\t')    #Start of FFT Indicator
                output.write('FFTr' + '\t\t')       #FFTr 
                output.write('FFTi' + '\n')         #FFTi

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t') #bin number
                output.write(str(np.uint16(x[1])) + '\t') #Last FFT Indicator
                output.write(str(np.uint16(x[2])) + '\t') #End of FFT Indicator
                output.write(str(np.uint16(x[3])) + '\t') #Start of FFT Indicator
                output.write(str(np.int32(x[4])) + '\t') #FFTr 
                output.write(str(np.int32(x[5])) + '\n') #FFTi 
    return

def save_ram(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('OpCode' + '\t') #OpCode
                output.write('Address' + '\t') #RAM Address 
                output.write('Data' + '\n') #Data
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint8(x[0]) & 0xff, '02X') + '\t') #OpCode
                output.write(format(np.uint16(x[1]) & 0xfff, '03X') + '\t') #RAM Address
                output.write(format(np.uint64(x[2]) & 0xffffffffffffffff, '016X') + '\n') #Data 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('OpCode' + '\t') #OpCode
                output.write('Address' + '\t') #Ram Address 
                output.write('Data' + '\n') #Data 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint8(x[0])) + '\t') #OpCode
                output.write(str(np.uint16(x[1])) + '\t') #RAM Address
                output.write(str(np.uint64(x[2])) + '\n') #Data 
    return

def save_xspectra(out_array, out_path, out_type):

    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('SBin' + '\t') #bin number
                output.write('Comp' + '\t\t') #Compressed result 
                output.write('Uncomp' + '\n') #Uncompressed result
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t') #bin number
                output.write(format(np.uint16(x[1]) & 0xffff, '04X') + '\t') #Compressed Result 
                output.write(format(np.uint64(x[2]) & 0xffffffffffffffff, '016X') + '\n') #Uncompressed Result
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('SBin' + '\t') #Bin number
                output.write('Comp' + '\t') #Compressed result 
                output.write('Uncomp' + '\n') #Uncompressed result 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t') #Bin number
                output.write(str(np.int32(x[1])) + '\t') #Compressed result
                output.write(str(np.int64(x[2])) + '\n') #Uncompressed result 
    return


def saveall(out_array, out_path, out_type):
    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('Bytes 1 & 2' + '\t') 
                output.write('Bytes 3 & 4' + '\t')  
                output.write('Bytes 5 & 6' + '\t')  
                output.write('Bytes 7 & 8' + '\t')  
                output.write('Bytes 9 &10' + '\t') 
                output.write('Bytes 11&12' + '\n') 
        
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint16(x[0]) & 0xffff, '04X') + '\t')
                output.write(format(np.uint16(x[1]) & 0xffff, '04X') + '\t') 
                output.write(format(np.uint16(x[2]) & 0xffff, '04X') + '\t') 
                output.write(format(np.uint16(x[3]) & 0xffff, '04X') + '\t') 
                output.write(format(np.uint16(x[4]) & 0xffff, '04X') + '\t')
                output.write(format(np.uint16(x[5]) & 0xffff, '04X') + '\n') 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'

        if not(os.path.exists(out_name)):
        #set up headers if file doesn't already exist
            with open(out_name, 'a') as output:
                output.write('Bytes 1 & 2' + '\t') 
                output.write('Bytes 3 & 4' + '\t')  
                output.write('Bytes 5 & 6' + '\t')  
                output.write('Bytes 7 & 8' + '\t')  
                output.write('Bytes 9 &10' + '\t') 
                output.write('Bytes 11&12' + '\n') 

        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint16(x[0])) + '\t')
                output.write(str(np.uint16(x[1])) + '\t')
                output.write(str(np.uint16(x[2])) + '\t')
                output.write(str(np.uint16(x[3])) + '\t')
                output.write(str(np.uint16(x[4])) + '\t')
                output.write(str(np.uint16(x[5])) + '\n')  
    return


def saveascsv(fname, adds, outputfolder='output'):
    import csv
    with open(outputfolder+'/'+fname, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adds)

def andbytes(abytes, bbytes):
    val = bytes([a & b for a, b in zip(abytes[::-1], bbytes[::-1])][::-1])
    return val