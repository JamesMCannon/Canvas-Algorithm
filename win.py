import numpy as np
import matplotlib.pyplot as plt

from saveas import save_output_txt

# ---------------------------- Get Hanning Window Coeffs ------------------------------
def get_win(nFFT, show_plots=False, save_output='both', out_folder='output'):
    
    # hanning window according to IDL func (asymmetric)

    win_out = [((2**16)-1)*(0.5 - (0.5)*np.cos(2* np.pi * k/ nFFT)) for k in range(nFFT)]
    win = [round(w) for w in win_out] # make int window
    win = np.array(win)

    if show_plots:
        plt.plot(np.arange(0, len(win)), win)
        plt.title('Input Window')
        plt.show()
        plt.close()
    
    if save_output:
        out_path = out_folder+'/window'
        save_output_txt(win, out_path, save_output, 'u-16')

    return win
# ------------------------------------------------------------------------------------