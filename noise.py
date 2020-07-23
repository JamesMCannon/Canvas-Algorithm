import numpy as np

# this is gaussian
# shot and Johnson noise are gaussian

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

for R in res:
    vrms = Johnson_noise(R, 0.3e3, 40e3)
    vsd = rms2sd(vrms, 0.3e3, 40e3)
    vsd_sum += vsd

print('total thermal noise spectral density', vsd_sum/(1e-9), ' nV/m -rtHz')

vsd_currentR1 = shot_noise(R1)
vsd_currentR2 = shot_noise(R3) # probs dont need this

print('total v spectral density from current noise', 1e9 * (vsd_currentR1 + vsd_currentR2), ' nV/m -rtHz')

vsd_sum = vsd_sum + vsd_currentR1 + vsd_currentR2 + 2.9e-9

print('total noise spectral density', vsd_sum/(0.4 * 1e-9), ' nV/m -rtHz')