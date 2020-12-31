import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt

# ------------------------------- Rebin for CANVAS fbins ----------------------------
# the center fs are for the 512 length FFT and the fbins are either from canvas bins or tx bins 
def rebin_canvas(acc_p, n_acc, fbins, center_freqs, show_plots=False, save_output='both', out_folder='output'): 

    avg_pwr = np.zeros((np.shape(acc_p)[0], np.shape(acc_p)[1], len(fbins)//2))
    for ci, c in enumerate(acc_p):
        for pi,p in enumerate(c): 

            # loop through canvas bins
            for fbins_ind in range(0, len(fbins), 2):
                # current canvas bin
                current_bin = (fbins[fbins_ind], fbins[fbins_ind+1])
                save_ind = fbins_ind//2

                # store power with freq inside the current canvas bin
                newbin_power = [] 

                # loop through fft bins and see if contained in current canvas bin
                for ff_ind, ff_val in enumerate(center_freqs): # match FPGA
                    if ff_val >= current_bin[0] and ff_val < current_bin[1]:
                        # append power value to list for new canvas bin
                        newbin_power.append(p[ff_ind-2])

                # this step AVERAGES the power by summing and dividing by the # of bins and # of accummulated ffts
                avg_pwr[ci][pi][save_ind] = np.floor(sum(newbin_power)/(len(newbin_power)*n_acc))
                # break at the last canvas bin
                if fbins_ind > len(fbins) - 4: 
                    break
    
        if show_plots:
            plt.plot(np.log10(avg_pwr[0]),'.')
            plt.title('avg power -- first')
            plt.show()
            plt.close()

        if save_output:
            for ap in avg_pwr[ci]:
                out_path = out_folder+'/channel'+str(ci)+'_avg'
                save_output_txt(ap, out_path, save_output, 'u-64')

    return avg_pwr # return size i x j where i = accum and j = len of fbins
# ------------------------------------------------------------------------------------