# trying to simulate filter design
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyvalfromroots

"""
# specs
rparray = np.linspace(0.01,0.5,100)
fcarray = np.linspace(42e3,50e3,10) # Hz
fs = 88e3 # Hz
N = 8 # order ? 

Gparray = -rparray
Gs = -60 # dB
print(fcarray)
def chebypoly(N, f, fc):
    if np.abs(f / fc) <= 1:
        C = np.cos(N * np.arccos(f / fc))
    else: 
        C = np.cosh(N * np.arccosh(f / fc))

    return C

for fc in fcarray:
    poles = []

    for Gp in Gparray:
        n = 1/(np.arccosh(fs / fc)) * np.arccosh(np.sqrt((10**(-Gs/10) - 1) / (10**(-Gp/10) - 1)))
        poles.append(n)

    plt.plot(rparray, poles, label = fc)
plt.axhline(7)
plt.axhline(6.75)

plt.axvline(0.2)
plt.axvline(0.4)
#plt.legend()
plt.show()

z, p, k = signal.cheby1(N, 0.1, 50e3 * 2 * np.pi, 'low', analog=True, output='zpk')
print(z, p, k)
w, h = signal.freqs(b, a)

plt.semilogx(w/(2*np.pi), 20 * np.log10(abs(h)))

plt.title('Chebyshev Type I frequency response')

plt.xlabel('Frequency')

plt.ylabel('Amplitude [dB]')

plt.margins(0, 0.1)

plt.grid(which='both', axis='both')


plt.show()
# We can either increase  fc  or decrease the ripple in the passband.
"""
"""

# start over
rps = np.linspace(0.01, 0.5, 100)
eps = np.sqrt(10**(rps/10) - 1)
sb = 60
A = 10**(sb / 20)

fs = 88e3
fp = np.linspace(42e3, 50e3, 10)

omegas = fs * 2 * np.pi
omegap = fp * 2 * np.pi

for op in omegap:
    N = np.arccosh(np.sqrt((A**2 - 1) / eps**2)) / np.arccosh(omegas / op)
    #plt.plot(rps, N, label='op')
#plt.show()

# final selections
N = 8
rps = 0.3
eps = np.sqrt(10**(rps/10) - 1)
op = 45e3 * 2 * np.pi

def find_s(k, N, eps):
    sigmak = -np.sin((2*k - 1) * np.pi/ (2*N)) * np.sinh((1/N) * np.arcsinh(1/eps))
    omegak = np.cos((2*k - 1) * np.pi/ (2*N)) * np.cosh((1/N) * np.arcsinh(1/eps))
    sk = sigmak + complex(0,1)*omegak
    return sk

poles = []
# we want 4 second order stages 
# omg i think this is right
for k in range(2, N + 1, 2):
    sk = find_s(k, N, eps)
    poles.append(sk)

R1R3 = [pole.real**2 + pole.imag**2 for pole in poles]

normconst = [1/x for x in R1R3]

C4 = 1e-9
# lets set R's equal for now

for normind, norm in enumerate(normconst):
    print(normind)
    F1 = 2 * np.abs(poles[normind].real) / norm
    # solve for R
    # F1 = C4 * (2*R) / (R**2)
    R = 2 * C4 / F1
    # solve for C2
    # F2 = C2 * C4 / (R**2)
    C2 = R**2 * norm / C4


# we'll end up adjusting them more, but we need to overall transfer function to plot
b, a = signal.cheby1(N, 0.1, 50e3 * 2 * np.pi, 'low', analog=True)

tf = 1
K = 1 / np.sqrt(1 + eps**2)

w = np.linspace(0, 100e3 * np.pi * 2, 100)
for normind, norm in enumerate(normconst):
    s = w / op
    tf = tf * norm * (s**2 + 2 * np.abs(poles[normind].real) * s + R1R3[normind])
    #plt.semilogx(w /(2*np.pi), K/tf)

w, h = signal.freqs(b, a)

plt.semilogx(w/(2*np.pi), 20 * np.log10(abs(h)))

#plt.show()

# for each stage

new_w = np.linspace(0, 100e3 * np.pi * 2, 100)
for nw in new_w: 
    denom = 0
    j = complex(0,1)
    for i in range(N+1):
        currentn = N-1
        denom += a[i]*(j*nw)**currentn
    num = b * (j*nw)**N
    tf = num / denom
    print(tf)

"""
"""
N = 8
rp = 0.1
eps = np.sqrt(10**(rp/10) - 1)
fp = 43e3
op = fp * np.pi * 2

# overall transfer function
z,p,k  = signal.cheby1(N, rp, op, 'low', analog=True, output = 'zpk')
warray = np.linspace(0,100e3*2*np.pi,1000)

zeta = [1 / ( - 2 * op * pole.real / (pole.real**2 + pole.imag**2)) for pole in p]
print(zeta)
tflist = []
hlist = []
# overall tf for each w
for w in warray:
    s = 1j * w
    num = polyvalfromroots(s, z)
    den = polyvalfromroots(s, p)
    h = k * num/den
    hlist.append(h)

hlist = np.array(hlist)
plt.semilogx(warray / (np.pi *2), 20 * np.log10(abs(hlist)))

w, h = signal.freqs_zpk(z,p,k)
#plt.semilogx(w / (np.pi *2), 20 * np.log10(abs(h)))
#plt.show()


"""
# good god

def freqcut(R,C):
    denom = 1
    for res, cap in zip(R, C):
        denom *= np.sqrt(res*cap)

    fc = 1/(2*np.pi*denom)

    return fc

# ideal
q1 = 0.593
q2 = 1.183
q3 = 2.453
q4 = 8.082
qs = [q1, q2, q3, q4]

# ideal 
f1 = 16.409e3
f2 = 27.743e3
f3 = 38.435e3
f4 = 44.47e3

C = np.array([1, 1.5, 2, 2.2, 3.3, 4.7, 5, 5.6, 6.8])

q1list = []
myq = q1
pq = 1
for c2 in C:
    for c1 in C:
        q = np.sqrt(c2 / c1) / 2
        cq = np.abs(q - myq)
        q1list.append(cq)
        #if np.abs(q - myq) < pq:
        #    print(np.abs(q - myq), c1, c2)
        pq = np.abs(q - myq)
#print(min(q1list))


q2list = []
myq = q2
pq = 1
for c2 in C:
    for c1 in C:
        q = np.sqrt(c2 / c1) / 2
        cq = np.abs(q - myq)
        q2list.append(cq)
        #if np.abs(q - myq) < pq:

        pq = np.abs(q - myq)


q3list = []
myq = q3
pq = 1
for c2 in C:
    c2 *= 10
    for c1 in C:
        q = np.sqrt(c2 / c1) / 2
        cq = np.abs(q - myq)
        q3list.append(cq)
       # if np.abs(q - myq) < pq:
           # print(np.abs(q - myq), c1, c2)
        pq = np.abs(q - myq)



q4list = []
myq = q4
pq = 1
for c2 in C:
    c2 *= 100
    for c1 in C:
        q = np.sqrt(c2 / c1) / 2
        cq = np.abs(q - myq)
        q4list.append(cq)
        #if np.abs(q - myq) < pq:
           # print(np.abs(q - myq), c1, c2)
        pq = np.abs(q - myq)



C11 = 1.0e-9
C12 = 1.5e-9
C21 = 1.0e-9
C22 = 5.6e-9
C31 = 2.0e-9
C32 = 47e-9
C41 = 2.2e-9
C42 = 560e-9

R1 = np.linspace(1, 10e3, 100000)
flist = []
for r in R1:
    fp = freqcut([r,r], [C11, C12])
    flist.append((np.abs(f1-fp)))
print(min(flist))
print(R1[flist.index(min(flist))])
print(R1[flist.index(min(flist))]/2)

R1 = np.linspace(1, 6e3, 100000)
flist = []
for r in R1:
    fp = freqcut([r,r], [C21, C22])
    flist.append((np.abs(f1-fp)))
print(min(flist))
print(R1[flist.index(min(flist))])
print(R1[flist.index(min(flist))]/2)


R1 = np.linspace(1, 2e3, 100000)
flist = []
for r in R1:
    fp = freqcut([r,r], [2.2e-9, C32])
    flist.append((np.abs(f1-fp)))
print(min(flist))
print(R1[flist.index(min(flist))])
print(R1[flist.index(min(flist))]/2)


R1 = np.linspace(1, 1e3, 100000)
flist = []
for r in R1:
    fp = freqcut([r,r], [C41, C42])
    flist.append((np.abs(f1-fp)))
print(min(flist))
print(R1[flist.index(min(flist))])
print(R1[flist.index(min(flist))]/2)