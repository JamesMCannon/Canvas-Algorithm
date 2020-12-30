"""

script to solve for Ez using elements of spectral matrix
using the fsolve feature, which solves nonlinear equations by
approximating them as linear eqns

"""

import numpy as np
from scipy.optimize import fsolve

# define set of equations, each line is the decomposed form of each
# off-diagonal elements of the 5x5 spectral matrix Q
# rearranged to solve for 0
# Q elements are known and should be defined below

# bxr indicates B real component in the x direction
# bxi indicate B imaginary component in the x direction

def f(z):
    bxr, bxi, byr, byi, bzr, bzi, exr, exi, eyr, eyi = z

    # define complex elements of Q
    Q12 = np.array([])
    Q13 = np.array([])
    Q14 = np.array([])
    Q15 = np.array([])
    Q23 = np.array([])
    Q24 = np.array([])
    Q25 = np.array([])
    Q34 = np.array([])
    Q35 = np.array([])
    Q45 = np.array([])

    f1 = (bxr * byr + bxi * (-byi)) + (bxr * (-byi) + bxi * byr) - Q12[0] - Q12[1]
    f2 = (bxr * bzr + bxi * (-bzi)) + (bxr * (-bzi) + bxi * bzr) - Q13[0] - Q13[1]
    f3 = (bxr * exr + bxi * (-exi)) + (bxr * (-exi) + bxi * exr) - Q14[0] - Q14[1]
    f4 = (bxr * eyr + bxi * (-eyi)) + (bxr * (-eyi) + bxi * eyr) - Q15[0] - Q15[1]
    f5 = (byr * bzr + byi * (-bzi)) + (byr * (-bzi) + byi * bzr) - Q23[0] - Q23[1]
    f6 = (byr * exr + byi * (-exi)) + (byr * (-exi) + byi * exr) - Q24[0] - Q24[1]
    f7 = (byr * eyr + byi * (-eyi)) + (byr * (-eyi) + byi * eyr) - Q25[0] - Q25[1]
    f8 = (bzr * exr + bzi * (-exi)) + (bzr * (-exi) + bzi * exr) - Q34[0] - Q34[1]
    f9 = (bzr * eyr + bzi * (-eyi)) + (bzr * (-eyi) + bzi * eyr) - Q35[0] - Q35[1]
    f10 = (exr * eyr + exi * (-eyi)) + (exr * (-eyi) + exi * eyr) - Q45[0] - Q45[1]

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

# solve the system with initial guesses
q = fsolve(f,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print('Re(Bx) =', q[0], '\n'
      'Im(Bx) =', q[1], '\n'
      'Re(By) =', q[2], '\n'
      'Im(By) =', q[3], '\n'
      'Re(Bz) =', q[4], '\n'
      'Im(Bz) =', q[5], '\n'
      'Re(Ex) =', q[6], '\n'
      'Im(Ex) =', q[7], '\n'
      'Re(Ey) =', q[8], '\n'
      'Im(Ey) =', q[9], '\n')
