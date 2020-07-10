# testing fc for 7pole CH

import numpy as np
import matplotlib.pyplot as plt 

# create function to find fc of each stage

def freqcut(R,C):
    denom = 1
    for res, cap in zip(R, C):
        denom *= np.sqrt(res*cap)

    fc = 1/(2*np.pi*denom)

    return fc

# ideal 
f1 = 16.409e3
f2 = 27.743e3
f3 = 38.435e3
f4 = 44.47e3


# caps to choose from
C1 = 1e-9
C2s2 = 3900e-12
C2s3 = 0.02e-6
C2s3_2 = 0.022e-6
C2s4 = 0.22e-6

# potential R values
R1s1 = 12.7e3
R1s2 = np.array(np.linspace(2.5e3, 4.5e3, 100))
R2s2 = np.array(np.linspace(2.5e3, 4.5e3, 100))
R1s3 = np.array(np.linspace(750, 950, 100))
R2s3 = np.array(np.linspace(950, 1.15e3, 100))
R1s4 = np.array(np.linspace(100, 300, 100))
R2s4 = np.array(np.linspace(150, 350, 100))


# get fc for each stage
    
fc2difflast = 1
for r1 in R1s2:
    for r2 in R2s2:
        fcs2 = freqcut([r1, r2], [C1, C2s2])
        fc2diff = np.abs(fcs2 - fc2)
        if fc2difflast > fc2diff:
            print('r1s2 ', r1, 'r2s2 ', r2)
            fc2difflast = fc2diff

fc3difflast = 1
for r1 in R1s3:
    for r2 in R2s3:
        fcs3 = freqcut([r1, r2], [C1, C2s3])
        fc3diff = np.abs(fcs3 - fc3)
        if fc3difflast > fc3diff:
            print('r1s3 ', r1, 'r2s3 ', r2)
            fc3difflast = fc3diff

fc4difflast = 100
for r1 in R1s4:
    for r2 in R2s4:
        fcs4 = freqcut([287, r2], [C1, C2s4])
        fc4diff = np.abs(fcs4 - fc4)
        if fc4difflast > fc4diff:
            print('r1s4 ', 287, 'r2s4 ', r2)
            fc4difflast = fc4diff

# 30k 43k
# 33k 38k
# 900 1.07
# 287 and.. ??

fcs2test1 = freqcut([3e3, 4.3e3], [C1, C2s2])
# ^ final selection, but saving the other in case
fcs2test2 = freqcut([3.3e3, 3.8e3], [C1, C2s2])

fcs3test1 = freqcut([900, 1.07e3], [C1, C2s3])
fcs3test2 = freqcut([900, 1.09e3], [C1, C2s3])
# ^ final selection but save the other in case

#print(np.abs(fc3 - fcs3test1))
#print(np.abs(fc3 - fcs3test2))

# final for s4 = 287 and 210

# TUNING WAS SUCCESSSSSSFULLLLL