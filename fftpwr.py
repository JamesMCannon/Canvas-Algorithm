import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt

# function to take spectra and xspectra power
# break down the complex arrays to make it easier to compare to the fpga
# input is for a single FFT output

# we shouldnt have to round either of these bc the input is int

# ---------------------------- Compute Power of Spectra -------------------------------
def fft_spec_power(real_data, imag_data, show_plots=False, save_output='both', out_folder='output'):

    spec_pwr = np.zeros_like(real_data)

    for k, (cr, ci) in enumerate(zip(real_data, imag_data)):
        for ind, (r,i) in enumerate(zip(cr, ci)):
            r = np.array(r)
            i = np.array(i)

            sp = r**2 + i**2

            if show_plots:
                plt.plot(np.log10(sp),'.')
                plt.title('spectra power')
                plt.show()
                plt.close()

            if save_output:
                out_path = out_folder+'/channel'+str(k)+'_spectra'
                save_output_txt(sp, out_path, save_output, 'u-64')

            spec_pwr[k][ind][:] = sp

    return spec_pwr
# -------------------------------------------------------------------------------------

# ---------------------------- Compute Power of XSpectra -------------------------------
def fft_xspec_power(c1_real_data, c1_imag_data, c2_real_data, c2_imag_data, show_plots=False, save_output='both', out_folder='output'):

    xspec_pwr_r = np.zeros_like(c1_real_data)
    xspec_pwr_i = np.zeros_like(c1_imag_data)

    # diff channel: R = real1*real2 + imag1*imag2 , I = real1*imag2 - real2*imag1
    for ind, (r1, i1, r2, i2) in enumerate(zip(c1_real_data, c1_imag_data, c2_real_data, c2_imag_data)):
        xspec_pwr_r[ind] = r1 * r2 + i1 * i2
        xspec_pwr_i[ind] = r1 * i2 - r2 * i1

    if show_plots:
        plt.plot(np.log10(xspec_pwr_r),'.',label='real')
        plt.plot(np.log10(xspec_pwr_i),'.',label='imag')
        plt.title('xspectra power')
        plt.legend()
        plt.show()
        plt.close()

    if save_output:
        out_path = out_folder+'/channel'+str(ci)+'_xspectra_real'
        save_output_txt(xspec_pwr_r, out_path, save_output, 'u-64')
        out_path = out_folder+'/channel'+str(ci)+'_xspectra_imag'
        save_output_txt(xspec_pwr_i, out_path, save_output, 'u-64')

    return xspec_pwr_r, xspec_pwr_i
# -------------------------------------------------------------------------------------