# Calculate CANVAS capacitance requirement
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib import cm

import pyglow
from skyfield.api import EarthSatellite
from sgp4.api import Satrec, WGS72
from skyfield.api import load, wgs84

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

# constants
qe = 1.6e-19
me = 9.11e-31
mi = 2000*me
kb = 1.38e-23
e0 = 8.854e-12
u0 = 4*np.pi*1e-7
c = np.sqrt(1/e0/u0)

# antenna / spacecraft / orbit parameters
a = .002/2               # radius, m 
L = 0.80                  # total length, m 
velocity = 7600           # m/s
antenna_sa = 2*np.pi*a*L  # m^2

# environment parameters
ne_min = 0.8e10 # Ne m^-3
ne_max = 0.5e12 # Ne m^-3
Te_min = 1500   # K
Te_max = 3000   # K

# ------------------- Find Debye Length ---------------------
lambda_d_min = np.sqrt((e0*kb*Te_min)/(ne_max*qe**2))
lambda_d_max = np.sqrt((e0*kb*Te_max)/(ne_min*qe**2))
print('min debye length m', lambda_d_min)
print('max debye length m', lambda_d_max) 

# --------------------- Find sheath capacitance -------------
C_s_min = 2*np.pi*e0*(L/2) / np.log(lambda_d_max/a)
C_s_max = 2*np.pi*e0*(L/2) / np.log(lambda_d_min/a)
print('min sheath cap pF', C_s_min*1e12)
print('max sheath cap pF', C_s_max*1e12)

# -------------------- Find sheath resistance ---------------
# electron current exceeds emitted photoelectron current
# negative potential (positive sheath)

# electron temperature in volts
ue_min = kb*Te_min/qe # V
ue_max = kb*Te_max/qe # V

# photoelectrton current estimate
Ip_min = 20e-6 * antenna_sa # A
Ip_max = 50e-6 * antenna_sa # A

# ram ion current -- assuming O+ ions only
Ii_min = qe*ne_min*velocity*antenna_sa/2
Ii_max = qe*ne_max*velocity*antenna_sa/2

# sheath resistance
R_s_min = ue_min/(Ip_max+Ii_max)
R_s_max = ue_max/(Ip_min+Ii_min)
print('min sheath res MOhm', R_s_min/1e6)
print('max sheath res MOhm', R_s_max/1e6)

# -------------------- Crossover frequency ---------------
fc_min = 1/(2*np.pi*R_s_max*C_s_max)
fc_max = 1/(2*np.pi*R_s_min*C_s_min)
print('cross over freq (shadow) kHz', fc_min/1e3)
print('cross over freq (sunlit) kHz', fc_max/1e3)

# --------------------------------------  Plot maps ---------------------------------------------
# load orbit propagator
satrec = Satrec()
ts = load.timescale()

# find mean motion
G = 6.67430e-11 # G
earth_mass = 5.972e24 #kg
r_earth = (6371+500)*1e3 #m
n = np.sqrt(G*earth_mass / r_earth**3)*60

# create orbit for CANVAS
satrec.sgp4init(
    WGS72,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    1,               # satnum: Satellite number
    21011.05731259,       # epoch: days since 1949 December 31 00:00 UT
    0.00011311,      # bstar: drag coefficient (/earth radii)
    0.00003644, # ndot: ballistic coefficient (revs/day)
    0.0,             # nddot: second derivative of mean motion (revs/day^3)
    0.0011442,       # ecco: eccentricity
    1.38614573, # argpo: argument of perigee (radians)
    0.9005899, # inclo: inclination (radians)
    4.90100322, # mo: mean anomaly (radians)
    n, # no_kozai: mean motion (radians/minute)
    1.402417433, # nodeo: right ascension of ascending node (radians)
)

# create sat object using orbit
sat = EarthSatellite.from_satrec(satrec, ts)

# set a start date and empty arrays to fill
st_date = datetime(2020,1,1)
ne = []
te = []
daynight = []
lats = []
lons = []

alt = 500 # km

# for earth rotation
eph = load('de421.bsp')

min_to_do = 60
# propagate every minute
for n in range(0,min_to_do,1):
    nm = timedelta(minutes=n)
    cdate = st_date + nm
    t = ts.utc(cdate.year, cdate.month, cdate.day, cdate.hour, cdate.minute, 0)
    
    # get the satellite pos
    geocentric = sat.at(t)
    sunlit = sat.at(t).is_sunlit(eph)
    subpoint = wgs84.subpoint(geocentric)
    
    # get ne and te data
    pt = pyglow.Point(cdate, subpoint.latitude.degrees, subpoint.longitude.degrees, alt)
    pt.run_iri()

    # save data
    ne.append(pt.ne * 100**3) # convert to m^-3
    te.append(pt.Te)
    daynight.append(sunlit)
    lats.append(subpoint.latitude.degrees)
    lons.append(subpoint.longitude.degrees)

    if n % (min_to_do/10) == 0:
        print('minutes is', n, 'of ', min_to_do)

print('done propagating')

# break into night and day
lats_day = [lats[i] for i in range(len(lats)) if daynight[i]==True]
lons_day = [lons[i] for i in range(len(lats)) if daynight[i]==True]
ne_day = [ne[i] for i in range(len(lats)) if daynight[i]==True]
te_day = [te[i] for i in range(len(lats)) if daynight[i]==True]

lats_night = [lats[i] for i in range(len(lats)) if daynight[i]==False]
lons_night = [lons[i] for i in range(len(lats)) if daynight[i]==False]
ne_night = [ne[i] for i in range(len(lats)) if daynight[i]==False]
te_night = [te[i] for i in range(len(lats)) if daynight[i]==False]

# quick func to find fc for plotting
def find_fc(ne_atpos, te_atpos):
    # constants
    qe = 1.6e-19
    me = 9.11e-31
    mi = 2000*me
    kb = 1.38e-23
    e0 = 8.854e-12
    u0 = 4*np.pi*1e-7
    c = np.sqrt(1/e0/u0)

    # antenna / spacecraft / orbit parameters
    a = .002/2               # radius, m 
    L = 0.80                  # total length, m 
    velocity = 7600           # m/s
    antenna_sa = 2*np.pi*a*L  # m^2

    lambda_d = np.sqrt((e0*kb*te_atpos)/(ne_atpos*qe**2))
    C_s = 2*np.pi*e0*(L/2) / np.log(lambda_d/a)

    ue = kb*te_atpos/qe # V
    Ip = 35e-6 * antenna_sa # A - avg
    Ii = qe*ne_atpos*velocity*antenna_sa/2
    R_s = ue/(Ip+Ii)

    fc = 1/(2*np.pi*R_s*C_s)

    return fc/1e3 

#pt = pyglow.Point(cdate, -45, 90, 500)
#pt.run_iri()  
#print(find_fc(pt.ne*1e3, pt.Te)) 

# use cartopy plot
projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
                dict(map_projection=projection))

fig = plt.figure()
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(2, 1),
                axes_pad=0.6,
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.2,
                cbar_size='3%',
                label_mode='') 

for i, ax in enumerate(axgr):
    ax.coastlines()
    ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
    ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylim([-55, 55])

H, xedges, yedges = np.histogram2d(lons_day, lats_day,  bins=50, range=None, normed=None, weights=None, density=None)
X, Y = np.meshgrid(xedges, yedges)

ne_grid_day = np.zeros_like(H)
te_grid_day = np.zeros_like(H)
fc_grid_day = np.zeros_like(H)

for check_pos, (check_lon, check_lat) in enumerate(zip(lons_day,lats_day)):
    for ii, (prev_lon, cur_lon) in enumerate(zip(xedges, xedges[1:])):
        if prev_lon <= check_lon <= cur_lon:
            saveii = ii
    for kk, (prev_lat, cur_lat) in enumerate(zip(yedges, yedges[1:])):
        if prev_lat <= check_lat <= cur_lat:
            savekk = kk
    ne_grid_day[saveii, savekk] += ne_day[check_pos]
    te_grid_day[saveii, savekk] += te_day[check_pos]

for j in range(np.shape(H)[0]):
    for k in range(np.shape(H)[1]):
        ne_grid_day[j,k] = ne_grid_day[j,k] / H[j,k]
        te_grid_day[j,k] = te_grid_day[j,k] / H[j,k]

        fc_grid_day[j,k] = find_fc(ne_grid_day[j,k], te_grid_day[j,k])

#p = axgr[0].pcolormesh(X, Y, fc_grid_day.T, cmap = 'plasma',vmin=10,vmax=200)
levels = np.linspace(10,100,num=9)
p = axgr[0].contourf(X[:-1,:-1], Y[:-1,:-1], fc_grid_day.T, levels, cmap=cm.coolwarm, extend='max')


# repeat for night
H, xedges, yedges = np.histogram2d(lons_night, lats_night,  bins=50, range=None, normed=None, weights=None, density=None)
X, Y = np.meshgrid(xedges, yedges)

ne_grid_night = np.zeros_like(H)
te_grid_night = np.zeros_like(H)
fc_grid_night = np.zeros_like(H)


for check_pos, (check_lon, check_lat) in enumerate(zip(lons_night,lats_night)):
    for ii, (prev_lon, cur_lon) in enumerate(zip(xedges, xedges[1:])):
        if prev_lon <= check_lon <= cur_lon:
            saveii = ii
    for kk, (prev_lat, cur_lat) in enumerate(zip(yedges, yedges[1:])):
        if prev_lat <= check_lat <= cur_lat:
            savekk = kk
    ne_grid_night[saveii, savekk] += ne_night[check_pos]
    te_grid_night[saveii, savekk] += te_night[check_pos]

for j in range(np.shape(H)[0]):
    for k in range(np.shape(H)[1]):
        ne_grid_night[j,k] = ne_grid_night[j,k] / H[j,k]
        te_grid_night[j,k] = te_grid_night[j,k] / H[j,k]

        fc_grid_night[j,k] = find_fc(ne_grid_night[j,k], te_grid_night[j,k])

#p = axgr[1].pcolormesh(X, Y, fc_grid_night.T, cmap = 'plasma',vmin=20,vmax=120)
p = axgr[1].contourf(X[:-1,:-1], Y[:-1,:-1], fc_grid_night.T, levels, cmap=cm.coolwarm, extend='max')

# formatting
axgr.cbar_axes[0].colorbar(p)
axgr[0].set_title('Sunlit')
axgr[1].set_title('Shadow')
#plt.savefig('avg_fc.png')
plt.close()

# plot capacitive gain region
cl = np.linspace(10,50,num=100)
cs = 6.
cg = cs/(cs+cl)
plt.plot(cl,cg)
plt.hlines(0.20, 10, 24, colors='r', linestyles='dashed',label='cl 24pF')
plt.hlines(0.15, 10, 34, colors='g', linestyles='dashed',label='cl 34pF')
plt.xlabel('load capacitance pF')
plt.ylabel('capacitive gain')
plt.legend()
plt.title('Capacitve Gain vs Load Capacitance for fc < 40kHz')
plt.savefig('cgain.png')
plt.close()