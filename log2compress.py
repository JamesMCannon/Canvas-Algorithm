import numpy as np 
import matplotlib.pyplot as plt 
import math
from saveas import save_output_txt

# ------------------------- log2 compression ------------------------------------
def spec_compress(avg_pwr, channel_num=0, show_plots=False, save_output='both', out_folder='output'):
    cmprs_pwr = []
    for iv in avg_pwr:
        if iv == 0:  # if 0, compressed is 0
            cmprs_pwr.append(0)
        else: # handling around 0.5 is odd, sometimes this will be 1 off
            cmprs = math.log2(iv)*64
            cmprs_pwr.append(round(cmprs))

    if save_output:
        out_path = out_folder+'/channel'+str(channel_num)+'_cmprs'
        save_output_txt(cmprs_pwr, out_path, save_output, 'u-16') # shouldnt be 64 anymore....

    return cmprs_pwr
# ------------------------------------------------------------------------------------

# ------------------------- log2 compression ------------------------------------
def xspec_compress(avg_pwr, coefficient="r", show_plots=False, save_output='both', out_folder='output'):
    cmprs_pwr = []
    for iv in avg_pwr:
        if iv == 0:  # if 0, compressed is 0
            cmprs_pwr.append(0)
        elif iv < 0: 
            iv = np.abs(iv)
            ov = round(math.log2(iv)*32)

            # convert to binary
            ov = str(bin(ov))

            # esaier to work w srtings
            ov = ov.replace('0b', '')
            while len(ov) < 11:
                ov = '0'+ ov

            # add 1 for negative to leftmost position
            ov = '1' + ov 
            # cap at 12
            ov = ov[:12]
            # convert back!
            ov = int(ov, 2)
            cmprs_pwr.append(ov)
        elif iv>0:
            cmprs_pwr.append(round(math.log2(iv)*32))
        else:
            print("Invalid value: ", iv)

    if save_output:
        out_path = out_folder+'/channel_'+coefficient+'_cmprs'
        save_output_txt(cmprs_pwr, out_path, save_output, 'u-16') # shouldnt be 64 anymore....

    return cmprs_pwr
# ------------------------------------------------------------------------------------