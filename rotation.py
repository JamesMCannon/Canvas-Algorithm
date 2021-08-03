import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

deg2rad = np.pi/180

# def XYZ
X = [1,0,0]
Y = [0,1,0]
Z = [0,0,1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0,1],[0,0],[0,0],'orange',label='X')
ax.plot([0,0],[0,1],[0,0],'g',label='Y')
ax.plot([0,0],[0,0],[0,1],'purple',label='Z')

#ax.plot([0,np.cos(45*deg2rad)],[0,0],[0,np.sin(45*deg2rad)],'r',label='a')
#ax.plot([0,np.cos(120*deg2rad)*np.cos(45*deg2rad)],[0,np.sin(120*deg2rad)*np.cos(45*deg2rad)],[0,np.sin(45*deg2rad)],'b',label='b')
#ax.plot([0,np.cos(240*deg2rad)*np.cos(45*deg2rad)],[0,np.sin(240*deg2rad)*np.cos(45*deg2rad)],[0,np.sin(45*deg2rad)],'y',label='c')


scm = np.array([[np.cos(35.26*deg2rad),np.cos(120*deg2rad)*np.cos(35.26*deg2rad),np.cos(240*deg2rad)*np.cos(35.26*deg2rad)],
  [0, np.sin(120*deg2rad)*np.cos(35.26*deg2rad), np.sin(240*deg2rad)*np.cos(35.26*deg2rad)],
  [np.sin(35.26*deg2rad), np.sin(35.26*deg2rad), np.sin(35.26*deg2rad)]])

r = np.linalg.inv(scm)
print(scm)
print(r)
rot = np.matmul(r,scm)

ax.plot([0,rot[0][0]],[0,rot[1][0]],[0,rot[2][0]],'r',label='a')
ax.plot([0,rot[0][1]],[0,rot[1][1]],[0,rot[2][1]],'b',label='b')
ax.plot([0,rot[0][2]],[0,rot[1][2]],[0,rot[2][2]],'y',label='c')


plt.legend()
plt.show()
plt.close()