import numpy as np 
import matplotlib.pyplot as plt 
import math
from saveas import save_output_txt

# ------------------------- log2 compression ------------------------------------
def log2compress(avg_pwr, show_plots=False, save_output='both', out_folder='output'):

    cmprs_pwr = np.zeros_like(avg_pwr)

    for i, ap in enumerate(avg_pwr):
        cmprs_pwr[i][:] = round(math.log2(ap) * 64)

    if save_output:
        out_path = out_folder+'/channel'+str(ci)+'_cmprs'
        for rp in rebin_pwr:
            save_output_txt(rp, out_path, save_output, 'u-64') # shouldnt be 64 anymore....

# ------------------------------------------------------------------------------------