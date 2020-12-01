# import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hanning
from scipy.io import loadmat
import scipy
import os
import datetime as dt
import collections

# import constants
c = 2.998e8 # m/s 

# ------------------------------- 2's COMP ---------------------------------------
def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value
# --------------------------------------------------------------------------------


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

    # unpack
    bx_data = data[0]
    by_data = data[1]

    n_samples = int(sample_len * fs_vlf)

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


# ---------------------------- Get Hanning Window Coeffs ------------------------------
def get_win(nFFT): # input = number of FFT points
    # hanning window according to IDL func (asymmetric)

    win_out = [((2**16)-1)*(0.5 - (0.5)*np.cos(2* np.pi * k/ nFFT)) for k in range(nFFT)]
    win = [int(w) for w in win_out] # make int window
    win = np.array(win)

    return win
# -------------------------------------------------------------------------------------

# ---------------------------- Compute Power of Spectra --------------------------------
def power_spectra(c):

    # real^2 + imag^2 = power of spectra term
    Ps = [np.real(fch)**2 + np.imag(fch)**2 for fch in c] 

    return Ps
# -------------------------------------------------------------------------------------


# ---------------------------- Compute Power of XSpectra -------------------------------
def power_xspectra(c1, c2):

    Pxs_r = []
    Pxs_i = []

    # diff channel: R = real1*real2 + imag1*imag2 , I = real1*imag2 - real2*imag1
    Pxs_r = [np.real(fch) * np.real(fchx) + np.imag(fch) * np.imag(fchx) for fch, fchx in zip(c1, c2)]
    Pxs_i = [np.real(fch) * np.imag(fchx) - np.real(fchx) * np.imag(fch) for fch, fchx in zip(c1, c2)]

    return Pxs_r, Pxs_i
# -------------------------------------------------------------------------------------


# ----------------------- Average Frequency Domain Data in Time ------------------------
def time_avg(P, nFFT, fs, nsec): # input power array

    # scalar representing number of time points for one second
    accumulate_t = int(nsec * fs/(nFFT/2))

    # loop through each time point
    P_avg = []

    # for every chunk of time points
    for t in range(0, len(P), accumulate_t):
        
        # accumulate for one second and save it
        P_seg = P[t:accumulate_t+t]
        P_avg.append(sum(P_seg) / len(P_seg))

    return P_avg # return size i x j where i = seconds and j = nFFT
# ------------------------------------------------------------------------------------


# ------------------------------- Rebin for CANVAS fbins ----------------------------
def rebin_canvas(P, fbins, center_freqs): # input power vec, canvas fbins, fft fbins
    # create an empty array to store rebinned power array
    rebinned_power = []

    # loop through each averaged second of the power array
    for p_sec in P:
        # save all rebinned freq for that time point
        rebinned_tp = [] 

        # loop through canvas bins
        for fbins_ind in range(0, len(fbins), 2):
            # current canvas bin
            current_bin = (fbins[fbins_ind], fbins[fbins_ind+1])

            # store power with freq inside the current canvas bin
            newbin_power = [] 

            # loop through fft bins and see if contained in current canvas bin
            for ff_ind, ff_val in enumerate(center_freqs):
                
                if ff_val >= current_bin[0] and ff_val < current_bin[1]:
                    # append power value to list for new canvas bin
                    newbin_power.append(p_sec[ff_ind])
            
            # add up the values contained in canvas bin and average
            rebinned_tp.append(sum(newbin_power)/len(newbin_power))

            # break at the last canvas bin
            if fbins_ind > len(fbins) - 4: 
                break

        # add final list of rebinned power for that time point
        rebinned_power.append(rebinned_tp)

    return rebinned_power # return size i x j where i = seconds and j = len of fbins
# ------------------------------------------------------------------------------------