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
data_start = dt.datetime(int(bx_data['start_year']), int(bx_data['start_month']), int(bx_data['start_day']), int(bx_data['start_hour']), int(bx_data['start_minute']), int(bx_data['start_second']))

# get data -- 10 min of data? 
bx_data = bx_data['data']
by_data = by_data['data']

# create a timevec for the data
data_dt = 100e3 / len(bx_data)
data_dt = dt.timedelta(microseconds=data_dt*1e6) # convert time int in data to microseconds

timevec = []
for i, i_data in enumerate(bx_data):
    timevec.append(data_start+(data_dt*(i+1)))
    print(i)
print('finished')
# convert to a dataframe
time_vec = pd.to_datetime(time_vec)
d_bx = {'time': time_vec, 'data': bx_data}
d_by = {'time': time_vec, 'data': by_data}

df_bx = pd.DataFrame(d_bx)
df_by = pd.DataFrame(d_by)

# data is 100kHz, but canvas sampling is 2^17 (131072 Hz) -- resample data
# use pandas resampling func to resample an interpolate - using polynomial order 2
upsampled = series.resample('D')
interpolated = upsampled.interpolate(method='spline', order=2)

# finally, convert to 16 bit int for input time series data
bx_array = np.array()
by_array = np.array()

bx_list = [np.int16(bxi) for bxi in bx_data]
bx_list = [np.int16(bxi) for bxi in bx_data]

bx = np.array(bx_list)
by = np.array(by_list)
# ---------------- checkpoint -------------