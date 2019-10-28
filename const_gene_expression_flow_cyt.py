from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm
from scipy.special import comb
from math import exp, log, sqrt
from matplotlib.patches import Ellipse
from numba import jit
from matplotlib.ticker import FormatStrFormatter


comm = mpi.COMM_WORLD
rank = comm.rank
print(comm.Get_size())
print(rank)

# %% Model
k_off = 0.5
k_on = 0.1
k_r = 50.0
gamma = 1.0
theta = np.array([k_off, k_on, k_r, gamma])

n_cells = 1000
n_t = 100
t_meas = np.linspace(1, 10, n_t)


SM = [[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]]
X0 = [[2, 0, 0]]
P0 = [1.0]
S0 = [0.0]


def dprop_t_factory(i):
    def dprop_t(t, out):
        out[:] = 0.0
        out[i] = 1.0
    return dprop_t


dprop_t_list = []
for i in range(0, 4):
    dprop_t_list.append(dprop_t_factory(i))
dprop_sparsity = np.eye(4, dtype=np.intc)


def prop_t(t, out):
    out[:] = theta[:]


def prop_x(reaction, X, out):
    if reaction == 0:
        out[:] = X[:, 0]
        return None
    if reaction == 1:
        out[:] = X[:, 1]
        return None
    if reaction == 2:
        out[:] = X[:, 1]
        return None
    if reaction == 3:
        out[:] = X[:, 2]
        return None


init_bounds = np.array([2, 2, 100])

# %%
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(np.array(X0), np.array(P0), [np.array(S0)] * 4)
solutions = solver.SolveTspan(t_meas, 1.0e-4)

# %%
# Parameters for the noise models
alpha = 0.8
sigmax = 0.1
mubg = 1
sigmabg = 2

y = np.linspace(0, 500, 100)
sigmatot = np.sqrt(sigmax**2 + sigmabg**2)

def flowcyt_weight(x, fout):
    C = np.sqrt(2*np.pi*sigmatot*sigmatot)
    fout[:] = np.exp(-(y[:] - alpha*x[2] - mubg)**2/(2*sigmatot*sigmatot))/C

intensity_pdfs = []
for i in range(0, n_t):
    intensity_pdfs.append()