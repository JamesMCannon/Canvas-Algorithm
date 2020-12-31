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
                    output.write(format(np.int16(x) & 0xffff, '04X') + '\n')
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
