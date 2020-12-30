import numpy as np 


def psd2rms(psd, f1, f2, m):
    bw = f2 - f1
    rms = np.sqrt(psd * bw) 
    rms = rms # return rms voltage for antenna
    return rms

# inan
highi = 100
lowi = 1

# santolik
highs = 10**(-2.1)
#10**(-1) # in mV ??? 

# parrot
highp = 1000

# nemec
highn = 100
lown = 0.1

fl = 300
fh = 40e3
ma = 0.8

superbolt = 10**(-1.2)


print(psd2rms(highi,fl, fh, ma) * 1e-3, psd2rms(highs,fl, fh, ma), psd2rms(highp,fl, fh, ma) * 1e-3, 'mV')
print(psd2rms(lowi, fl, fh, ma) * 1e-3, 'mV')

print(psd2rms(highn,fl, fh, ma) * 1e-3, 'mV')
print(psd2rms(lown,fl, fh, ma) * 1e-3, 'mV')

print(psd2rms(superbolt,fl, fh, ma), 'mV', 'super')

print(psd2rms(10**(-9),fl, fh, ma), 'nT', 'super')