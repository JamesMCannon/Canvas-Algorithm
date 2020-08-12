import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# this is gaussian
# shot and Johnson noise are gaussian
import seaborn as sns
sns.set(style="whitegrid")
# lets stick with a barebones estiamte and see if we need to improve from there
def Johnson_noise(R, f1, f2):
    B = f2 - f1 # Hz
    T = 300 # K 
    k = 1.38e-23 # J/K
    vrms = np.sqrt(4 * k * R * B * T)
    return vrms 

def rms2sd(rms, f1, f2):
    B = f2 - f1 # Hz
    sd = rms / np.sqrt(B)
    return sd

def shot_noise(R):
    isd = 6e-15 # A/rt Hz
    vsd = isd * R
    return vsd
R1 = 3.3e3
R2 = 1e6
R3 = 47
res = [R1, R2, R3]

vsd_sum = 0

f1 = 300
f2 = 40e3

for R in res:
    if R < 1e5:
        vrms = Johnson_noise(R, f1, f2)
        vsd = rms2sd(vrms, f1, f2)
        vsd_sum += vsd

antenna_l = 0.8 
thermalsum = vsd_sum

farray = np.linspace(f1, f2, 1000)
thermalnoise = 1e6 * (thermalsum / antenna_l) * np.ones(len(farray))
shotnoise = 1e6 * 10**(-7.2) * np.ones(len(farray))

vsd_currentR1 = shot_noise(R1)
vsd_currentR2 = shot_noise(R2)
vsd_currentR3 = shot_noise(R3) # probs dont need this
currentnoise = np.ones(len(farray)) * 1e6 * (vsd_currentR1 + vsd_currentR2 + vsd_currentR3) / antenna_l

whitenoise = 7e-9
fcorner = 1.2e3
vnoise_op = []

for f in farray:
    if f < fcorner:
        vnoise_op.append(1e6 * whitenoise * np.sqrt(fcorner) * np.sqrt(1/f) / antenna_l)
    else:
        vnoise_op.append(1e6 * whitenoise / antenna_l)

totalnoise = [v + t + c for v,t,c in zip(vnoise_op, thermalnoise, currentnoise)]


fig, ax = plt.subplots(1, 1)
#plt.semilogx(farray, thermalnoise, '--', label='thermal noise')
plt.loglog(farray, shotnoise, '--', label='shot noise')
#plt.semilogx(farray, currentnoise, '--', label='current noise')
#plt.semilogx(farray, vnoise_op, '--', label='vnoise op amp')
plt.loglog(farray, totalnoise, '--', label='total receiver noise')
plt.loglog(farray, 1 * np.ones(len(farray)), '--', label='CANVAS min. req.')
#plt.loglog(farray, 10**(-7.2) * 10e6 * np.ones(len(farray)), '--', label='min whistler amplitude Inan et al., 2007')

plt.text(farray[10],shotnoise[10] + 10**(-2.1),'shot noise')
plt.text(farray[10],totalnoise[10]+ 10**(-2.5),'total receiver noise')
plt.text(farray[10],1.2,'CANVAS min. req.')
plt.text(farray[50],10,'lightning whistlers')

#plt.legend(loc='center right', fontsize = 8)
plt.title('Noise Estimate for CANVAS Antenna')
plt.ylabel('log(uV/m/rt-Hz)')
plt.xlabel('Hz')

rect = patches.Rectangle((300,10**(-7.2) * 10e6), 40e3, 10**(-5)*10e6 - 10**(-7.2) * 10e6, linewidth=1,alpha = 0.3,facecolor='b')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()