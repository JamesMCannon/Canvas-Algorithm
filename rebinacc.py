import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt

# FPGA first removes bins 0,1 and after 331, so FFT are now 330 in length
# let's do this in one step on the power values 

# ------------------------- rebin like the FPGA ------------------------------------
def rebin_likefpga(pwr, show_plots=False, save_output=False, out_folder='output'):
    
    rebin_pwr = np.zeros((np.shape(pwr)[0],np.shape(pwr)[1],330))
    for ci, c in enumerate(pwr):
        for pi, p in enumerate(c):
            rp = p[2:332]

            if show_plots:
                plt.plot(np.log10(rebin_pwr),'.')
                plt.title('rebin power')
                plt.show()
                plt.close()

            if save_output:
                out_path = out_folder+'/channel'+str(ci)+'_rebin'
                save_output_txt(rp, out_path, save_output, 'u-64')
            
            rebin_pwr[ci][pi][:] = rp

    return rebin_pwr 
# ------------------------------------------------------------------------------------

# ------------------------- acc like the FPGA ------------------------------------
def acc_likefpga(rebin_pwr, n_acc, show_plots=False, save_output='both', out_folder='output'):
    
    new_rebin = []
    pad = len(rebin_pwr[0])%n_acc
    for ci, c in enumerate(rebin_pwr):
        if pad!=0:
            for i in range(n_acc-pad):
                c = list(c)
                c.append(np.zeros(np.shape(rebin_pwr)[2]))
        new_rebin.append(c)

    acc_pwr = np.zeros((np.shape(new_rebin)[0], np.shape(new_rebin)[1]//n_acc,  np.shape(new_rebin)[2]))
    for ci, c in enumerate(new_rebin):
        for ti, t in enumerate(range(0, len(c), n_acc)):
            for p in range(0, n_acc):
                acc_pwr[ci][ti] += c[p+t]

        if show_plots:
            plt.plot(np.log10(acc_pwr),'.')
            plt.title('acc power')
            plt.show()
            plt.close()

        if save_output:
            out_path = out_folder+'/channel'+str(ci)+'_acc'
            for ap in acc_pwr[ci]:
                save_output_txt(ap, out_path, save_output, 'u-64')

    return acc_pwr 
# ------------------------------------------------------------------------------------