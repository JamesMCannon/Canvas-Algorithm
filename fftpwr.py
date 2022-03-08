import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt

# function to take spectra and xspectra power
# break down the complex arrays to make it easier to compare to the fpga
# input is for a single FFT output

# we shouldnt have to round either of these bc the input is int

# ---------------------------- Compute Power of Spectra -------------------------------
def fft_spec_power(real_data, imag_data, channel_num=0, show_plots=False, save_output='both', out_folder='output'):
    spec_pwr = np.zeros_like(real_data,dtype=np.int64)

    for ind, (r,i) in enumerate(zip(real_data, imag_data)):
        r = np.array(r,dtype=np.int64)
        i = np.array(i,dtype=np.int64)
        sp = r**2 + i**2

        if save_output:
            out_path = out_folder+'/channel'+str(channel_num)+'_spectra'
            save_output_txt(sp, out_path, save_output, 'u-64')

        spec_pwr[ind] = sp

    if show_plots:
        plt.plot(np.log10(spec_pwr[0]))
        plt.title('spectra power')
        plt.show()
        plt.close()

    return spec_pwr
# -------------------------------------------------------------------------------------

# ---------------------------- Compute Power of XSpectra -------------------------------
def fft_xspec_power(c1_real_data, c1_imag_data, c2_real_data, c2_imag_data, channel_nums=[0,1], show_plots=False, save_output='both', out_folder='output'):

    xspec_pwr_r = np.zeros_like(c1_real_data,dtype=np.int64)
    xspec_pwr_i = np.zeros_like(c1_imag_data,dtype=np.int64)

    # diff channel: R = real1*real2 + imag1*imag2 , I = real1*imag2 - real2*imag1
    for ind, (r1, i1, r2, i2) in enumerate(zip(c1_real_data, c1_imag_data, c2_real_data, c2_imag_data)):
        r1=np.array(r1,dtype=np.int64)
        i1=np.array(i1,dtype=np.int64)
        r2=np.array(r2,dtype=np.int64)
        i2=np.array(i2,dtype=np.int64)

        xspec_pwr_r[ind] = r1 * r2 + i1 * i2
        xspec_pwr_i[ind] = r2 * i1 - (r1 * i2)

    if show_plots:
        plt.plot(np.log10(xspec_pwr_r),label='real')
        plt.plot(np.log10(xspec_pwr_i),label='imag')
        plt.title('xspectra power')
        plt.legend()
        plt.show()
        plt.close()

    if save_output: # this IS signed!
        out_path = out_folder+'/channel'+str(channel_nums[0])+str(channel_nums[1])+'_xspectra_real_pwr'
        save_output_txt(xspec_pwr_r, out_path, save_output, 's-64')
        out_path = out_folder+'/channel'+str(channel_nums[0])+str(channel_nums[1])+'_xspectra_imag_pwr'
        save_output_txt(xspec_pwr_i, out_path, save_output, 's-64')

    return xspec_pwr_r, xspec_pwr_i
# -------------------------------------------------------------------------------------