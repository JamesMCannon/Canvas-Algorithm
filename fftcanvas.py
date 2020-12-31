import numpy as np 
import matplotlib.pyplot as plt

from saveas import save_output_txt

# ------------------------------------- CANVAS FFT alg -----------------------------------------
def canvas_fft(nFFT, fs, win, channels_td, overlap=True, show_plots=False, save_output='both', out_folder='output'):

    channels_fd_real = [] # store channels now in frequency domain
    channels_fd_imag = [] # store channels now in frequency domain

    for ci, c in enumerate(channels_td):

        c_fd_r = [] # store each segment of channel f domain
        c_fd_i = [] 

        # first, handle any channels that aren't nFFT*n points in length
        remainder = int(len(c)) % nFFT
        if remainder !=0: # remainder means not an integer multiple of nFFT
            need_zero = nFFT - remainder # this will be the # of missing points
            for i in range(need_zero):
                c.append(0)

        if overlap: # only setup for 50% overlap or no overlap....
            looplen = nFFT//2
        else:
            looplen = nFFT

        for i in range(0, len(c), looplen):
            cs_2 = c[i:i+nFFT]

            # this is handling the LAST FFT with overlap and padding with 0's
            if len(cs_2) != nFFT:
                cs_2.extend(np.zeros(nFFT//2))

            # mutitply elementwise by windowing func
            cs_2 = np.array(cs_2)
            cs_win = np.multiply(win, cs_2) # should be integer (with max 2^31-1) -- SIGNED 32 BIT

            # ---------------------------check win * input---------------------------------
            if i==0 and show_plots: # first FFT
                plt.plot(cs_win)
                plt.title('Input Window x Input Signal - First 1024')
                plt.show()
                plt.close()

            if save_output:
                out_path = out_folder+'/channel'+str(ci)+'_win'
                save_output_txt(cs_win, out_path, save_output, 's-32')
            # ----------------------------------------------------------------------------

            # take FFT
            cs_f = np.fft.fft(cs_win)

            # make it match IDL
            cs_f = cs_f / nFFT

            # convert real and imag to int
            # only need first half (match IDL/FPGA) -- signed 32 bit output
            cs_f_r = [round(np.real(c_r)) for c_r in cs_f[:nFFT//2]]
            cs_f_i = [round(np.imag(c_i)) for c_i in cs_f[:nFFT//2]]

            # ---------------------------check FFT ----------------------------------- 
            center_freqs = [fs/nFFT * ff for ff in np.arange(1, 513)]
            if i==0 and show_plots:
                plt.semilogy(center_freqs[0:nFFT//2], np.abs(cs_f[0:nFFT//2]), '-r')
                plt.title('FFT')
                plt.show()
                plt.close()
            
            if save_output:
                out_path = out_folder+'/channel'+str(ci)+'_fft_real'
                save_output_txt(cs_f_r, out_path, save_output, 's-32')
                out_path = out_folder+'/channel'+str(ci)+'_fft_imag'
                save_output_txt(cs_f_i, out_path, save_output, 's-32')
            # ---------------------------------------------------------------------------
            
            # save it
            c_fd_r.append(cs_f_r)
            c_fd_i.append(cs_f_i)

        # save the output for each channel - vector of 1024-pt FFTs
        channels_fd_real.append(c_fd_r)
        channels_fd_imag.append(c_fd_i)

    return channels_fd_real, channels_fd_imag
# ------------------------------------------------------------------------------------
