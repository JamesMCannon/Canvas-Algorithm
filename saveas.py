import numpy as np
import matplotlib.pyplot as plt
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
    return
# ------------------------------------------------------------------------------------

def save_FFT(out_array, out_path, out_type):

    #set up headers
    with open(out_name, 'a') as output:
            output.write('FBin' + '\t') #bin number
            output.write('FFTr' + '\t') #FFTr 
            output.write('FFTi' + '\n') #FFTi 
    
    #save data
    if out_type =='hex' or out_type =='both':
        out_name = out_path+'_hex.txt'
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(format(np.uint8(x[0]) & 0xff, '02X') + '\t') #bin number
                output.write(format(np.int16(x[1]) & 0xffff, '04X') + '\t') #FFTr 
                output.write(format(np.int16(x[2]) & 0xffff, '04X') + '\n') #FFTi 
    if out_type == 'int' or out_type == 'both':
        out_name = out_path+'_int.txt'
        with open(out_name, 'a') as output:
            for x in out_array:
                output.write(str(np.uint8(x[0])) + '\t') #bin number
                output.write(str(np.int16(x[1])) + '\t') #FFTr 
                output.write(str(np.int16(x[2])) + '\n') #FFTi 
    return

def saveascsv(fname, adds, outputfolder='output'):
    import csv
    with open(outputfolder+'/'+fname, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adds)