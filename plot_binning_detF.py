import numpy as np
import mpi4py.MPI as MPI
from chebpy import chebfun
import chebpy
from scipy.stats import logistic
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
from basic_units import radians, degrees, cos

with np.load('detFbinning.npz') as file:
    detF = file['detF']
    ABOUNDS = file['abounds']
    BBOUNDS = file['bbounds']
# np.savez('detFbinning.npz', detF = detF, abounds=ABOUNDS, bbounds=BBOUNDS)
NA = detF.shape[1]
NB = detF.shape[0]
ARGBOUNDS = [np.pi/6.0, 5*np.pi/12.0]

detF = np.log10(detF)
fig, ax = plt.subplots(1, 1)
fig.set_tight_layout(True)
acoos = np.logspace(ARGBOUNDS[0], ARGBOUNDS[1], NA)
bcoos = np.linspace(BBOUNDS[0], BBOUNDS[1], NB)
ax.contourf(acoos, bcoos, detF, xunits=radians)
ax.set_xlabel('arctan(a/4)')
ax.set_ylabel('b')
plt.show()
fig.savefig('binning_info_contours.pdf', bbox_inches='tight')
#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(20, 30)

X, Y = np.meshgrid(acoos, bcoos)

ax.plot_surface(X, Y, detF, color='orange')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('$\\log_{10}\\operatorname{det}(F)$')

plt.show()