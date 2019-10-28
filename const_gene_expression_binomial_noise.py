from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
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
t_meas = np.linspace(1, 10, 100)

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
alpha = 0.8 # success rate in binomial model
sigman = 0.8
mubg = 1
sigmabg = 2

def binom_weight(x, fout, p_success, n_max):
    fout[:] = binom.pmf(np.linspace(0, n_max, n_max+1), x[2], p_success[x[2]])

def compute_fim_exact(sens_sol):
    observed_marginal_sens = []
    observed_marginal_sens.append(sens_sol.Marginal(2))
    for iS in range(0,4):
        observed_marginal_sens.append(sens_sol.SensMarginal(iS, 2))
    fim = np.zeros((4,4))
    for iS1 in range(0,4):
        for iS2 in range(0,4):
            fim[iS1, iS2] = np.sum(
                   observed_marginal_sens[iS1+1]*observed_marginal_sens[iS2+1]/np.maximum(1.0e-16, observed_marginal_sens[0])
            )
    return fim

def compute_fim_binom(sens_sol):
    n_max = 200
    p_success = alpha*np.ones((n_max+1,))
    observed_marginal_sens = []
    for iS in range(-1,4):
        observed_marginal_sens.append(sens_sol.WeightedAverage(iS, n_max+1, lambda x, fout: binom_weight(x, fout, p_success, n_max)))
    fim = np.zeros((4,4))
    for iS1 in range(0,4):
        for iS2 in range(0,4):
            fim[iS1, iS2] = np.sum(
                   observed_marginal_sens[iS1+1]*observed_marginal_sens[iS2+1]/np.maximum(1.0e-16, observed_marginal_sens[0])
            )
    return fim
# %%

FIM_exact = np.zeros((len(t_meas), 4, 4))
FIM_binom = np.zeros((len(t_meas), 4, 4))
dopt_exact = np.zeros((len(t_meas,)))
dopt_binom = np.zeros((len(t_meas), ))
for i in range(0, len(t_meas)):
    FIM_exact[i, :, :] = n_cells*compute_fim_exact(solutions[i])
    FIM_binom[i, :, :] = n_cells*compute_fim_binom(solutions[i])
    dopt_exact[i] = np.linalg.det(FIM_exact[i, :, :])
    dopt_binom[i] = np.linalg.det(FIM_binom[i, :, :])

if rank == 0:
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_meas, dopt_binom, label='Binomial')
    ax.plot(t_meas, dopt_exact, label='Noise-free')
    ax.legend()
    plt.show()