import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt
from readFPGA import flatten

# FPGA first removes bins 0,1 and after 331, so FFT are now 330 in length
# let's do this in one step on the power values 

# ------------------------- rebin like the FPGA ------------------------------------
def rebin_likefpga(pwr, channel_num=0, show_plots=False, save_output=False, out_folder='output'):
    
    rebin_pwr = []
    for p in range(0,len(pwr),512):
        rp = pwr[2+p:332+p]

        

        if save_output:
            out_path = out_folder+'/channel'+str(channel_num)+'_rebin'
            save_output_txt(rp, out_path, save_output, 'u-64')
        
        rebin_pwr.append(rp)
    if show_plots:
            plt.plot(np.log10(rebin_pwr[0]))
            plt.title('rebin power')
            plt.show()
            plt.close()
    rebin_pwr = flatten(rebin_pwr) #from n x 330 to (n*330) x 1
    

    return rebin_pwr 
# ------------------------------------------------------------------------------------

# ------------------------- acc like the FPGA ------------------------------------
def acc_likefpga(rebin_pwr, n_acc, channel_num=0, show_plots=False, save_output='both', out_folder='output'):
    
    acc_f2 = np.zeros((len(rebin_pwr)//(n_acc*330),330))
    print(len(rebin_pwr)//(n_acc))
    for i in range(0,len(rebin_pwr)//(n_acc*330)):
        
        for k in range(330):
            argh = []
            for j in range(n_acc*i*330,n_acc*(i+1)*330,330):
                check = rebin_pwr[j:j+330]
                #print(i,j,k)
                argh.append(int(check[k]))
            mysum=0
            for ar in argh:
                mysum+=int(ar)
            thatval = mysum
            acc_f2[i][k] = thatval
    print(np.shape(acc_f2))
    if show_plots:
        plt.plot(np.log10(acc_f2[0]))
        plt.title('acc power')
        plt.show()
        plt.close()

    if save_output:
        out_path = out_folder+'/'+ 'channel' + str(channel_num) + '_acc'
        save_output_txt(flatten(list(acc_f2)), out_path, save_output, 'u-64')

    return flatten(list(acc_f2)) 
# ------------------------------------------------------------------------------------