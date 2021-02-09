import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt
from readFPGA import flatten

# ------------------------------- Rebin for CANVAS fbins ----------------------------
# the center fs are for the 512 length FFT and the fbins are either from canvas bins or tx bins 
def rebin_canvas(acc_p, n_acc, ci, fbins, center_freqs, show_plots=False, save_output='both', out_folder='output'): 
    all_avg_pwr = []
    for pi in range(0,len(acc_p),330): 
        avg_pwr = []
        p = acc_p[pi:pi+330]

        # loop through canvas bins
        for fbins_ind in range(0, len(fbins), 2):
            # current canvas bin
            current_bin = (fbins[fbins_ind], fbins[fbins_ind+1])

            # store power with freq inside the current canvas bin
            newbin_power = [] 

            # loop through fft bins and see if contained in current canvas bin
            for ff_ind, ff_val in enumerate(center_freqs): # match FPGA
                if ff_val >= current_bin[0] and ff_val < current_bin[1]:
                    # append power value to list for new canvas bin
                    newbin_power.append(p[ff_ind-2])

            # this step AVERAGES the power by summing and dividing by the # of bins and # of accummulated ffts
            avg_pwr.append(np.floor(sum(newbin_power)/(len(newbin_power)*n_acc)))
            #print(len(newbin_power))
            # break at the last canvas bin
            if fbins_ind > len(fbins) - 4: 
                break

        all_avg_pwr.append(avg_pwr)

    avg_pwr = flatten(all_avg_pwr)

    return avg_pwr # return size i x j where i = accum and j = len of fbins
# ------------------------------------------------------------------------------------