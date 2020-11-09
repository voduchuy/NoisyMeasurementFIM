from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
from pypacmensl.sensitivity.distribution import SensDiscreteDistribution
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


dprop_t_list = [dprop_t_factory(i) for i in range(4)]
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
def vec_add(v1: np.array, v2: np.array):
    v = np.zeros((max([len(v1), len(v2)])))
    v[0:len(v1)] = v1
    v[0:len(v2)] += v2
    return v

def weighted_vecadd(v: [np.array], weights: [float]):
    ans = weights[0]*v[0]
    for j in range(1, len(v)):
        ans = vec_add(ans, weights[j]*v[j])
    return ans

def temporally_distorted_distributions(exp_lambdas: [float], init_sens: SensDiscreteDistribution, integral_stepsize:
float, trunc_tol: float=1.0E-8):

    ell = min(exp_lambdas)
    B = 1.2*np.log(ell/trunc_tol)/ell
    nnodes_int = int(np.ceil(B / integral_stepsize))
    h = np.linspace(0, B, nnodes_int)

    comm = mpi.COMM_WORLD
    init_bounds = np.array([1, 1, 20])
    solver = SensFspSolverMultiSinks(comm)
    solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
    solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
    solver.SetInitialDist1(init_sens)
    solver.SetVerbosity(2)
    solutions = solver.SolveTspan(h, 1.0E-4)

    Y_distributions = {ell: [] for ell in exp_lambdas}
    Y_sensitivities = {ell: [] for ell in exp_lambdas}

    for ell in exp_lambdas:
        ps = [solutions[i].Marginal(2) for i in range(nnodes_int)]
        ws = ell*np.exp(-ell*h)
        Y_distributions[ell] = integral_stepsize*weighted_vecadd(ps, ws)

        sens_list = []
        for iS in range(0, 4):
            ss = [solutions[i].SensMarginal(iS, 2) for i in range(nnodes_int)]
            sens_list.append(integral_stepsize*weighted_vecadd(ss, ws))
        Y_sensitivities[ell] = sens_list

    return Y_distributions, Y_sensitivities

#%%
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_fsp = True
    else:
        run_fsp = False

    if run_fsp:
        t_outputs = np.linspace(0, 400, 401)
        comm = mpi.COMM_WORLD
        init_bounds = np.array([1, 1, 20])
        solver = SensFspSolverMultiSinks(comm)
        solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
        solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
        solver.SetInitialDist(np.array(X0), np.array([1.0]), np.array([0.0]*4))
        solver.SetVerbosity(2)
        undistorted_ps = solver.SolveTspan(t_outputs, 1.0E-6)

        k = 100
        exp_lambdas = [1.0, 1.0/5.0, 1.0/10.0]
        dists = {ell: {'p':[], 's':[]} for ell in exp_lambdas}

        for i in range(len(t_outputs)):
            print(f"Generating distorted distributions and sensitivities for t = {t_outputs[i]: .2e}")
            p_Y, S_Y = temporally_distorted_distributions(exp_lambdas, undistorted_ps[i], 0.1, 1.0E-7)

            for ell in exp_lambdas:
                dists[ell]['p'].append(p_Y[ell])
                dists[ell]['s'].append(S_Y[ell])

            if mpi.COMM_WORLD.Get_rank() == 0:
                np.savez("results/temporally_distorted_dists.npz", dists=dists, t_meas=t_outputs)

#%%
    with np.load("results/temporally_distorted_dists.npz", allow_pickle=True) as f:
        t_meas = f["t_meas"]
        dists = f["dists"][()]

    exp_lambdas = list(dists.keys())
    n_time = len(t_meas)
    n_par = len(dists[exp_lambdas[0]]['s'][0])
    temporal_fims = {}
    for ell in exp_lambdas:
        fim_temporal = np.zeros((n_time, n_par, n_par))
        for itime in range(n_time):
            p = dists[ell]['p'][itime]
            for ipar in range(n_par):
                si = dists[ell]['s'][itime][ipar]
                for jpar in range(0, ipar+1):
                    sj = dists[ell]['s'][itime][jpar]
                    fim_temporal[itime, ipar, jpar] = np.sum(si * sj / np.maximum(1.e-16, p))
            for ipar in range(0, n_par):
                for jpar in range(ipar + 1, n_par):
                    fim_temporal[ipar, jpar] = fim_temporal[jpar, ipar]
        temporal_fims[ell] = fim_temporal
    if mpi.COMM_WORLD.Get_rank() == 0:
        np.savez("results/temporal_fim.npz", temporal_fims=temporal_fims)