from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
#%%
k_off = 0.015
k_on = 0.05
k_r = 5
gamma = 0.05
#%%
t_meas = np.linspace(0, 400, 401)

theta = np.array([k_on, k_off, k_r, gamma])

SM = [[-1, 1, 0],
      [1, -1, 0],
      [0, 0, 1],
      [0, 0, -1]]

X0 = [[1, 0, 0]]
P0 = [1.0]
S0 = [0.0]


def dprop_t_factory(i: int):
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

#%%
def temporally_distorted_distributions(exp_lambda: float, X0: [[int]], t_outputs: [float], integral_stepsize: float,
                                       trunc_tol: float=1.0E-8):

    B = 1.2*np.log(exp_lambda/trunc_tol)/exp_lambda
    nnodes_int = int(np.ceil(B / integral_stepsize))
    h = np.linspace(0, B, nnodes_int)
    t_cme = np.unique(np.array([x + h for x in t_outputs]).flatten())
    t_idxs = {t_cme[i]: i for i in range(len(t_cme))}

    comm = mpi.COMM_WORLD
    init_bounds = np.array([1, 1, 20])
    solver = SensFspSolverMultiSinks(comm)
    solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
    solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
    solver.SetInitialDist(np.array(X0), np.array(P0), [np.array(S0)] * 4)
    solver.SetVerbosity(2)
    solutions = solver.SolveTspan(t_cme, 1.0e-8)

    Y_distributions = []
    Y_sensitivities = []

    def vec_add(v1: np.array, v2: np.array):
        v = np.zeros((max([len(v1), len(v2)])))
        v[0:len(v1)] = v1
        v[0:len(v2)] += v2
        return v

    for t in t_outputs:
        i = t_idxs[t]
        ptmp = exp_lambda*solutions[i].Marginal(2)
        for j in range(0, nnodes_int):
            i = t_idxs[t + h[j]]
            ptmp2 = exp_lambda*np.exp(-exp_lambda*h[j])*solutions[i].Marginal(2)
            ptmp = vec_add(ptmp, ptmp2)
        ptmp = integral_stepsize*ptmp
        Y_distributions.append(ptmp)

        sens_list = []
        for iS in range(0, 4):
            i = t_idxs[t]
            stmp = exp_lambda * solutions[i].SensMarginal(iS, 2)
            for j in range(0, nnodes_int):
                i = t_idxs[t + h[j]]
                stmp2 = exp_lambda * np.exp(-exp_lambda * h[j]) * solutions[i].SensMarginal(iS, 2)
                stmp = vec_add(stmp, stmp2)
            stmp = integral_stepsize*stmp
            sens_list.append(stmp)
        Y_sensitivities.append(sens_list)

    return Y_distributions, Y_sensitivities

#%%


if __name__ == "__main__":

    dists = {}
    t_outputs = np.linspace(0, 400, 401)
    k = 100

    for exp_lambda in [1.0, 1.0/5.0, 1.0/10.0]:
        dists[exp_lambda] = {'p': [], 's': []}
        for i in range(0, len(t_outputs), k):
            i1 = min(len(t_outputs), i+k)
            p_Y, S_Y = temporally_distorted_distributions(exp_lambda, X0, t_outputs[i:i1], 0.1)
            dists[exp_lambda]['p'].append(p_Y)
            dists[exp_lambda]['s'].append(S_Y)

    np.savez("results/temporally_distorted_dists.npz", dists=dists, t_meas = t_outputs)

