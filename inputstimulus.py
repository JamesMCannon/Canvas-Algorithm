# import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import os
import datetime as dt

from saveas import save_output_txt
from scipy.signal import chirp
from scipy import signal

# ---------------------------- Get VLF Data Table Mtn ------------------------------
def get_vlfdata(datadir): # directory of data
    # import input data from VLF rx on table mtn
    datafiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]

    # x data is the 000.mat file 
    for dfile in datafiles:
        if dfile[-5] == '0':
            bx_datafile = os.path.join(datadir, dfile)
        else:
            by_datafile = os.path.join(datadir, dfile)

    # load mat files
    bx_data = loadmat(bx_datafile)
    by_data = loadmat(by_datafile)
    
    return bx_data, by_data
# ------------------------------------------------------------------------------------

# -------------------------------- Resample VLF Data ---------------------------------
def resample(data, sample_len, fs_vlf, fs):

    # grab start time
    data_start = dt.datetime(int(bx_data['start_year']), int(bx_data['start_month']), 
    int(bx_data['start_day']), int(bx_data['start_hour']), int(bx_data['start_minute']), 
    int(bx_data['start_second']))

    bx_data = np.squeeze(bx_data['data'][:int(n_samples)])
    by_data = np.squeeze(by_data['data'][:int(n_samples)])

    # create a timevec for the data at current sample rate
    data_dt_vlf = dt.timedelta(microseconds=1e6/fs_vlf) # convert time delta to datetime obj.
    time_vec_vlf = [data_start+(data_dt_vlf*i) for i in range(int(n_samples))] # create time vec

    # create a timevec for the data at desired sampling freq.
    data_dt = dt.timedelta(microseconds=1e6/fs) # convert time delta to datetime obj. - NOT WORKING roundoff error
    time_vec = [data_start+(data_dt*i) for i in range(int(fs * n_samples / fs_vlf))] # create time vec

    # interpolate w a linear func 
    t_vlf = np.linspace(0, len(time_vec_vlf), num=len(time_vec_vlf), endpoint=True)
    t_fs = np.linspace(0, len(time_vec_vlf), num=len(time_vec), endpoint=True)

    f_x = scipy.interpolate.interp1d(t_vlf, bx_data)
    f_y = scipy.interpolate.interp1d(t_vlf, by_data)

    bx_a = f_x(t_fs)
    by_a = f_y(t_fs)

    # convert to 16 bit inputs - NO
    bx = [np.int16(x) for x in bx_a]
    by = [np.int16(y) for y in by_a]

    return bx, by
# ------------------------------------------------------------------------------------

# ------------------------------- Create a Test Signal ------------------------------- 
def test_signal(fs, sample_len, freq, amp, shift=0, channel_num=0, show_plots=False, save_output='both', out_folder='output'):

    # create time domain data
    samples = int(fs*sample_len)
    del_t = 1/fs
    t_vec = np.linspace(0, sample_len-del_t, num=samples)   # create time vec
    
    channels_td_raw = amp * np.sin(freq * 2 * np.pi * t_vec + shift)

    channels_td_rd =  [round(c,0) for c in channels_td_raw] # rounded to be nearest int

    channels_td = [int(c) for c in channels_td_rd] #cast to (16 bit) signed input

    if show_plots:
        plt_chk = int(len(channels_td))
        plt.plot(t_vec[:plt_chk], channels_td[:plt_chk])
        plt.plot(t_vec[:plt_chk], channels_td_raw[:plt_chk])
        plt.title('Input Signal - first 1024')
        plt.show()
        plt.close()

    # cast input (ints) to 16bit int represented in hex or integers
    if save_output:
        out_path = out_folder+'/channel'+str(channel_num)+'_input'
        save_output_txt(channels_td, out_path, save_output, 's-16')
    return channels_td
# ------------------------------------------------------------------------------------


# some other types of input signals we tried
def input_chirp(fs, sample_len, f0, f1, amp, show_plots=True, save_output='both', out_folder='output'):
    # create time domain data
    t_vec = np.linspace(0, sample_len, num=int(fs*sample_len))   # create time vec
    
    channels_td_raw = []
    channels_td_raw.append(amp * chirp(t_vec, f0=f0, f1=f1, t1=0.1, method='linear'))

    channels_td = [] # rounded to be 16 bit signed input
    for ctd in channels_td_raw:
        cx = [round(c,0) for c in ctd]
        channels_td.append(cx)

    if show_plots:
        plt_chk = 1024
        for ch in channels_td:
            plt.plot(t_vec[:plt_chk], ch[:plt_chk])
        plt.title('Input Signal - first 1024')
        plt.show()
        plt.close()

    # cast input (ints) to 16bit int represented in hex or integers
    for ci, c in enumerate(channels_td):
        if save_output:
            out_path = out_folder+'/chirp'
            save_output_txt(c, out_path, save_output, 's-16')
    return channels_td

# ------------------------------------------------------------------------------------

def white_noise(fs, sample_len, amp, show_plots=True, save_output='both', out_folder='output'):
    # create time domain data
    t_vec = np.linspace(0, sample_len, num=int(fs*sample_len))   # create time vec
    
    channels_td_raw = []

    noise = amp*np.random.normal(0, .1, t_vec.shape)
    channels_td_raw.append(noise)

    channels_td = [] # rounded to be 16 bit signed input
    for ctd in channels_td_raw:
        cx = [round(c,0) for c in ctd]
        channels_td.append(cx)

    if show_plots:
        plt_chk = 1024
        for ch in channels_td:
            plt.plot(t_vec[:plt_chk], ch[:plt_chk])
        plt.title('Input Signal - first 1024')
        plt.show()
        plt.close()

    # cast input (ints) to 16bit int represented in hex or integers
    for ci, c in enumerate(channels_td):
        if save_output:
            out_path = out_folder+'/white_noise'
            save_output_txt(c, out_path, save_output, 's-16')
    return channels_td