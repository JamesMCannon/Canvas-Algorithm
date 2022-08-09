import numpy as np
import matplotlib.pyplot as plt

# python functions to read FPGA input files (in hex)

# ---------------------------- 2's comp ---------------------------------------------
def twos_complement(hexstr,b):
    value = int(hexstr,16) # hex is base 16
    if value & (1 << (b-1)):
        value -= 1 << b
    return value
# ------------------------------------------------------------------------------------

# ---------------------------- read FPGA input ---------------------------------------
def read_FPGA_input(file, b=16, signed=True, show_plots=False):
    f = open(file, 'r')
    datalines = [line for line in f]
    if signed:
        fpga_in_data = [twos_complement(p,b) for p in datalines]
    else:
        fpga_in_data = [int(p,16) for p in datalines]

    f.close()

    if show_plots:
        plt.plot(fpga_in_data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(fpga_in_data))

    return fpga_in_data
# ------------------------------------------------------------------------------------

def read_FPGA_cmprs(file, line_n):
    f = open(file, 'r')
    datalines = [line for line in f]
    comp = datalines[line_n:]
    f.close()

    sign_mask = b'\x08\x00'
    mag_mask = b'\x07\xFF'

    comp_val=[]
    for i in range(len(comp)):
        hex_comp = bytes.fromhex(comp[i])
        sign = andbytes(hex_comp,sign_mask)
        comp_mag = int.from_bytes(andbytes(hex_comp,mag_mask),'big')
        if int.from_bytes(sign,'big')>0:
            comp_val.append(-comp_mag)
        else:
            comp_val.append(comp_mag)
            
    d1 = comp_val

    return d1

# ---------------------------- read INT input ---------------------------------------
def read_INT_input(file, show_plots=False):
    f = open(file, 'r')
    data = [int(line.strip('\n')) for line in f]
    f.close()

    if show_plots:
        plt.plot(data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(data))

    return data
# ------------------------------------------------------------------------------------

# ---------------------------- quick compare ---------------------------------------
def quick_compare(py_array, fp_array, vals, show_plots=False):
    py_array = np.array(py_array)
    fp_array = np.array(fp_array)

    diff = (py_array[:vals] - fp_array[:vals]) / py_array[:vals]
    
    if show_plots:
        plt.plot(diff)
        plt.show()
        plt.close()

    return diff
# ------------------------------------------------------------------------------------

def flatten(mylist):
    flat_list = [item for sublist in mylist for item in sublist]
    return flat_list

# ------------------------------------------------------------------------------------

def read_FPGA_fft_debug(file, b, signed):
    f = open(file, 'r')
    datalines = [line for line in f]
    
    fpga_data = {}
    save_di = 0
    count = 0
    for di,dl in enumerate(datalines):
        if dl[0] == 'F':
            if dl == 'FFT Stage 9 Input Samples\n' and di!=0:
                data_len = 256
            else:
                data_len = 258
            dl = dl.strip('\n')
            fpga_data[dl+str(count//10)] = {}
            headers = datalines[di+1].split()
            cd = datalines[di+2:di+data_len]
            cd_split = [c.split() for c in cd]
            cd_flat = flatten(cd_split)
            for hi, h in enumerate(headers):
                if h == 'WR':
                    h = 'WR(COS)'
                    headers.pop(8)
                that_data = [cd_flat[k] for k in range(hi,len(cd_flat),len(headers))]
                fpga_data[dl+str(count//10)][h] = that_data
            count += 1 

    print(fpga_data['FFT Stage 8 Input Samples23']['TF_INDEX'])
    
    if signed:
        fpga_in_data = [twos_complement(p,b) for p in datalines]
    else:
        fpga_in_data = [int(p,16) for p in datalines]

    f.close()

    if show_plots:
        plt.plot(fpga_in_data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(fpga_in_data))

    return fpga_in_data
    
# ------------------------------------------------------------------------------------ 

# ------------------------------------------------------------------------------------ 

def read_FPGA_input_lines(file, b, line_n, x, y, signed=True, show_plots=False):
    f = open(file, 'r')
    datalines = [line.split() for line in f]
    datalines = flatten(datalines)
    datalines = datalines[line_n:]
    if signed:
        fpga_in_data = [twos_complement(p,b) for p in datalines]
    else:
        fpga_in_data = [int(p,16) for p in datalines]

    f.close()

    if show_plots:
        plt.plot(fpga_in_data[:1024],'-')
        plt.show()
        plt.title(file)
        plt.close()

    print('reading FPGA input \n file length is: ', len(fpga_in_data))

    d1 = [fpga_in_data[n] for n in range(x,len(fpga_in_data),line_n)]
    d2 = [fpga_in_data[n] for n in range(y,len(fpga_in_data),line_n)]

    return d1, d2

def read_FPGA_xspectra(file, line_n):
    f = open(file, 'r')
    datalines = [line.split() for line in f]
    sbin=[];comp=[];uncomp=[]
    for r in datalines:
        sbin.append(r[0])
        comp.append(r[1])
        uncomp.append(r[2])
    sbin = sbin[line_n:]
    comp = comp[line_n:]
    uncomp = uncomp[line_n:]
    f.close()

    sign_mask = b'\x08\x00'
    mag_mask = b'\x07\xFF'

    comp_val=[]
    uncomp_val=[]
    for i in range(len(comp)):
        hex_comp = bytes.fromhex(comp[i])
        sign = andbytes(hex_comp,sign_mask)
        comp_mag = int.from_bytes(andbytes(hex_comp,mag_mask),'big')
        if int.from_bytes(sign,'big')>0:
            comp_val.append(-comp_mag)
            uncomp_val.append(-int(uncomp[i],16))
        else:
            comp_val.append(comp_mag)
            uncomp_val.append(int(uncomp[i],16))


    d1 = comp_val
    d2 = uncomp_val

    return d1, d2

# ------------------------------------------------------------------------------------ 

def read_FPGA_fft(file,b=32,header=True,signed=True):
    f = open(file,'r')
    re = []
    im = []
    datalines = [line.split() for line in f]
    if header:
        datalines = datalines[1:]
    for i in datalines:
        re.append(i[1])
        im.append(i[2])
    real = [twos_complement(p,b) for p in re]
    imaginary = [twos_complement(p,b) for p in im]
    return real,imaginary

def andbytes(abytes, bbytes):
    val = bytes([a & b for a, b in zip(abytes[::-1], bbytes[::-1])][::-1])
    return val
