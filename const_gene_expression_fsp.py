from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np

k_off = 0.001
k_on = 0.005
k_r = 0.5
gamma = 0.01

theta = np.array([k_on, k_off, k_r, gamma])

SM = [[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]]
X0 = [[1, 0, 0]]
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


init_bounds = np.array([1, 1, 20])
t_meas = np.linspace(0, 20*60, 20*60 + 1)

comm = mpi.COMM_SELF
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(np.array(X0), np.array(P0), [np.array(S0)] * 4)
solutions = solver.SolveTspan(t_meas, 1.0e-4)

rna_distributions = []
rna_sensitivities = []
for i in range(0, len(solutions)):
    rna_distributions.append(solutions[i].Marginal(2))
    sens_list = []
    for iS in range(0, 4):
        sens_list.append(solutions[i].SensMarginal(iS, 2))
    rna_sensitivities.append(sens_list)

np.savez('bursting_parameters.npz', kon=k_on, koff = k_off, kr= k_r, gamma=gamma)
np.savez('fsp_solutions.npz', rna_distributions=rna_distributions, rna_sensitivities=rna_sensitivities, t_meas=t_meas, allow_pickle=True)