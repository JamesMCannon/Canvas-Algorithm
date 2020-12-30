import numpy as np 
import matplotlib.pyplot as plt



import seaborn as sns
sns.set(style="whitegrid")

def addquad(uns):
    sm = 0
    for un in uns:
        sm+=un**2
    quad = np.sqrt(sm)
    return quad

def findu(ms):
    u = ms * 0.008 + 0.1
    return u

meas = [30.59, 30.05, 29.85, 29.40]

for ms in meas:
    nu = findu(ms)
    print(nu)
    uns = [0.274, nu]
    newu = addquad(uns)
    print((ms - 21.78), ' ', newu)

with open('capactiance.txt', 'r') as f:
    lines = f.readlines()

# remove spaces
lines = [line.replace(' ', '') for line in lines]
lines = [" ".join(line.split()) for line in lines]

lines = lines[:-1]
mymeas = []
myun = []

for line in lines: 
    mymeas.append(float(line))

with open('uncer.txt', 'r') as f:
    lines = f.readlines()

# remove spaces
lines = [line.replace(' ', '') for line in lines]
lines = [" ".join(line.split()) for line in lines]

for line in lines: 
    myun.append(float(line))

mymeas = np.asarray(mymeas)
myun = np.asarray(myun)

x = np.linspace(0, len(mymeas), len(mymeas))
plt.figure(figsize=(10,5))
plt.plot(x, mymeas, c='0.55')
plt.fill_between(x, mymeas-myun, mymeas+myun, alpha=0.3, facecolor='b')
plt.xlabel('seconds')
plt.ylabel('capacitance (pF)')

plt.show()
