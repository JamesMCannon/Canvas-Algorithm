import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt
from readFPGA import flatten

# ------------------------------- Rebin for CANVAS fbins ----------------------------
# the center fs are for the 512 length FFT and the fbins are either from canvas bins or tx bins 
def rebin_canvas(acc_p, n_acc, fbins, center_freqs, tx_bins=True, channel_num=0, show_plots=False, save_output='both', out_folder='output'): 
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
            #print(sum(newbin_power))
            avg_pwr.append(np.floor(sum(newbin_power)//(len(newbin_power)*n_acc)))
            # break at the last canvas bin
            if fbins_ind > len(fbins) - 4: 
                break

        all_avg_pwr.append(avg_pwr)

    if tx_bins==True:
        # parse text file with VLF TX canvas bins
        fname = 'CANVAS_fbins/tx_fbins.txt'                                 
        TX_fbins_str = np.genfromtxt(fname, dtype='str') 
        TX_fbins_cen = [(float(f[3:].replace(',','')))*1e3 for f in TX_fbins_str]
        TX_fbins_names = [TXn[:3] for TXn in TX_fbins_str]
        TX_fbins_dbl = [(f - 100., f + 100.) for f in TX_fbins_cen]
        
        fbins = flatten(TX_fbins_dbl)
        final_avg_pwr = []
        for pi in range(0,len(acc_p),330):
            avg_pwr = []
            p = acc_p[pi:pi+330]

            # loop through canvas bins
            for tx, fbins_ind in zip(TX_fbins_names, range(0, len(fbins), 2)):
                # current canvas bin
                current_bin = (fbins[fbins_ind], fbins[fbins_ind+1])
                # store power with freq inside the current canvas bin
                newbin_power = [] 

                # loop through fft bins and see if contained in current canvas bin
                for ff_ind, ff_val in enumerate(center_freqs): # match FPGA
                    if ff_val >= current_bin[0] and ff_val < current_bin[1]:
                        # append power value to list for new canvas bin
                        newbin_power.append(p[ff_ind-2])
                        #print(tx)
                    elif ff_val+35 >= current_bin[0] and ff_val < current_bin[1]:
                        # append power value to list for new canvas bin
                        newbin_power.append(p[ff_ind-2])
                        #print(tx)
                    elif ff_val >= current_bin[0] and ff_val-35 < current_bin[1]:
                        # append power value to list for new canvas bin
                        newbin_power.append(p[ff_ind-2])
                        #print(tx)

                # this step AVERAGES the power by summing and dividing by the # of bins and # of accummulated ffts
                if fbins_ind == 0 or fbins_ind == 16: # for HWU and NRK
                    avg_pwr.append(np.floor(sum(newbin_power)//(4*n_acc)))
                else:
                    avg_pwr.append(np.floor(sum(newbin_power)//(2*n_acc)))
                # break at the last canvas bin
                if fbins_ind > len(fbins) - 4: 
                    break
            
            # reorder packing
            tx_order = [1, 2, 3, 4, 5, 6, 7, 9, 0, 8]
            re_order_pwr = [avg_pwr[i] for i in tx_order]
            canvas_avg = all_avg_pwr[pi//330]
            for item in re_order_pwr:
                canvas_avg.append(item)
            final_avg_pwr.append(canvas_avg)
    else:
        final_avg_pwr = all_avg_pwr


    if show_plots:
        plt.plot(np.log10(final_avg_pwr[0]))
        plt.title('Average Power')
        plt.show()
        plt.close()
    avg_pwr = flatten(final_avg_pwr)

    if save_output:
        out_path = out_folder+'/channel'+str(channel_num)+'_avg'
        save_output_txt(avg_pwr, out_path, save_output, 'u-64')
    


    return avg_pwr # return size i x j where i = accum and j = len of fbins
# ------------------------------------------------------------------------------------

def fix_neg1(py_avg_pwr, fp_avg_pwr):
    # fix for negative 1 issue
    avg_pwr_fix = []
    for av,fv in zip(py_avg_pwr, fp_avg_pwr):
        if np.abs(av-fv) > 0:
            av = av + 1
        avg_pwr_fix.append(av)
    return avg_pwr_fix