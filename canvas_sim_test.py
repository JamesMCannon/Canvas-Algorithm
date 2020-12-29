# import statements
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.signal
from scipy.io import loadmat
import os
import pandas as pd
import datetime as dt
 
# import constants
c = 2.998e8 # m/s 

# import input data from VLF on table mtn
datadir = 'vlf_data'
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

# grab start time
data_start = dt.datetime(int(bx_data['start_year']), int(bx_data['start_month']), 
int(bx_data['start_day']), int(bx_data['start_hour']), int(bx_data['start_minute']), 
int(bx_data['start_second']))

# define param of input data
fs_vlf = 100e3
n_samples = 100e3 # user defined
data_len_vlf = n_samples / fs_vlf

bx_data = np.squeeze(bx_data['data'][:int(n_samples)])
by_data = np.squeeze(by_data['data'][:int(n_samples)])

# create a timevec for the data at current sample rate
data_dt_vlf = dt.timedelta(microseconds=1e6/fs_vlf) # convert time delta to datetime obj.
time_vec_vlf = [data_start+(data_dt_vlf*i) for i in range(int(n_samples))] # create time vec

# create a timevec for the data at desired sampling freq.
fs = 2**17
data_dt = dt.timedelta(microseconds=7.62939) # convert time delta to datetime obj. - nano sec to avoid roundoff error
time_vec = [data_start+(data_dt*i) for i in range(int(fs * n_samples / fs_vlf))] # create time vec

# interpolate w a linear func 
t_vlf = np.linspace(0, len(time_vec_vlf), num=len(time_vec_vlf), endpoint=True)
t_fs = np.linspace(0, len(time_vec_vlf), num=len(time_vec), endpoint=True)

f_x = scipy.interpolate.interp1d(t_vlf, bx_data)
f_y = scipy.interpolate.interp1d(t_vlf, by_data)

bx = f_x(t_fs)
by = f_y(t_fs)
print(data_dt*fs)

#plt.plot(time_vec_vlf, bx_data, '-', time_vec, bx, '-')
#plt.show()