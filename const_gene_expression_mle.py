from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot
import mpi4py.MPI as mpi
import numpy as np
from scipy.stats import binom
import pygmo

RANK = mpi.COMM_WORLD.Get_rank()
NPROCS = mpi.COMM_WORLD.Get_size()
np.random.seed(RANK)

k_off_true = 0.005
k_on_true = 0.001
k_r_true = 0.5
gamma_true = 0.01
p_success = 0.5
ncells = 100
init_bounds = np.array([1, 1, 20])
t_meas = np.linspace(20*60/4, 20*60, 4)
theta0 = np.array([k_on_true, k_off_true, k_r_true, gamma_true])
SM = np.array([[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]])
X0 = np.array([[1, 0, 0]])
P0 = np.array([1.0])

theta_lb = 0.001*theta0
theta_ub = 100*theta0
log10theta_lb = np.log10(theta_lb)
log10theta_ub = np.log10(theta_ub)

def BinomialNoiseMatrix(n_max, p_success):
    M = np.zeros((n_max + 1, n_max + 1))
    for j in range(0, n_max + 1):
            M[:, j] = binom.pmf(np.linspace(0, n_max, n_max+1), j, p_success)
    return M

n_max = 2000
C_binom = BinomialNoiseMatrix(n_max, p_success)


def propensity_factory(theta):
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
    return prop_t, prop_x

def simulate_data(theta):
    """Simulate distorted SmFish observations"""
    prop_t, prop_x = propensity_factory(theta0)
    ssa = SSASolver(mpi.COMM_SELF)
    ssa.SetModel(SM, prop_t, prop_x)
    data = []
    for t in t_meas:
        X = ssa.Solve(t, X0, ncells, send_to_root=True)
        y = np.random.binomial(X[:,2], p_success)
        data.append(SmFishSnapshot(y))
    return data

def solve_cme(log10_theta):
    theta = np.power(10.0, log10_theta)
    propensity_t, propensity_x = propensity_factory(theta)
    cme_solver = FspSolverMultiSinks(mpi.COMM_SELF)
    cme_solver.SetVerbosity(0)
    cme_solver.SetModel(SM, propensity_t, propensity_x)
    cme_solver.SetFspShape(constr_fun=None, constr_bound=init_bounds, exp_factors=np.array([0.0, 0.0, 0.0, 0.2]))
    cme_solver.SetInitialDist(X0, P0)
    cme_solver.SetUp()
    solutions = cme_solver.SolveTspan(t_meas, 1.0E-6)
    cme_solver.ClearState()
    return solutions

def neg_loglike_wrong(log10_theta, smfishdata):

    try:
        solutions = solve_cme(log10_theta)
    except:
        return 1.0E8

    ll = 0.0
    for i in range(0, len(t_meas)):
        ll = ll + smfishdata[i].LogLikelihood(solutions[i], np.array([2]))
    return -1.0 * ll

def neg_loglike_right(log10_theta, smfishdata):
    try:
        solutions = solve_cme(log10_theta)
    except:
        return 1.0E8

    ll = 0.0
    for i in range(0, len(t_meas)):
        xobs = smfishdata[i].GetStates()
        xobs = xobs.T
        pobs = smfishdata[i].GetFrequencies()
        prna = solutions[i].Marginal(2)
        xmax = len(prna) - 1
        py = C_binom[:, 0:xmax + 1] @ prna
        ll = ll + np.sum(pobs*np.log( np.maximum(py[xobs], 1.0e-16) ))
    return -1.0 * ll

class my_prob_wrong:
    def __init__(self):
        self.dim = 4

    def get_bounds(self):
        return (log10theta_lb, log10theta_ub)

    def fitness(self, dv):
        fhandle = open(f'joint_data_loglike_evals_loc_opt_{RANK}.txt', 'a')
        fhandle.write('log_theta = {0}'.format(str(dv)))
        ll = neg_loglike_wrong(dv, data)
        fhandle.write(f'll = {ll} \n')
        fhandle.close()
        return [ll]

    def gradient(self, x):
        return pygmo.estimate_gradient(lambda x: self.fitness(x), x)  # we here use the low precision gradient

class my_prob_right:
    def __init__(self):
        self.dim = 4

    def get_bounds(self):
        return (log10theta_lb, log10theta_ub)

    def fitness(self, dv):
        fhandle = open(f'correct_loglike_evals_{RANK}.txt', 'a')
        fhandle.write('log_theta = {0}'.format(str(dv)))
        ll = neg_loglike_right(dv, data)
        fhandle.write(f'll = {ll} \n')
        fhandle.close()
        return [ll]

    def gradient(self, x):
        return pygmo.estimate_gradient(lambda x: self.fitness(x), x)  # we here use the low precision gradient
#
POPULATION_SIZE = 1

# Compute fits with the correct likelihood function

num_trials = 10
fits_local = np.zeros((num_trials, 4))

for itrial in range(0, num_trials):
    data = simulate_data(theta0)
    prob = pygmo.problem(my_prob_right())
    pop = pygmo.population(prob)
    pop.push_back(np.log10(theta0))
    start_range0 = 0.01
    my_algo = pygmo.compass_search(max_fevals=1000, start_range=start_range0, stop_range=1.0E-5)
    algo = pygmo.algorithm(my_algo)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)
    fits_local[itrial, :] = pop.champion_x

if RANK == 0:
    fits_all = np.zeros((num_trials*NPROCS, 4))
    buffersizes = 4*num_trials*np.ones((NPROCS,), dtype=int)
    displacements = np.zeros((NPROCS,), dtype=int)
    displacements[1:] = np.cumsum(buffersizes[0:-1])
else:
    fits_all = None
    buffersizes = None
    displacements = None

mpi.COMM_WORLD.Gatherv(sendbuf=fits_local, recvbuf=[fits_all, buffersizes, displacements, mpi.DOUBLE], root=0)

if RANK == 0:
    np.savez("ge_mle_correct_fits.npz", thetas=fits_all)

# Compute fits with the incorrect likelihood function

num_trials = 10
fits_local = np.zeros((num_trials, 4))

for itrial in range(0, num_trials):
    data = simulate_data(theta0)
    prob = pygmo.problem(my_prob_wrong())
    pop = pygmo.population(prob)
    pop.push_back(np.log10(theta0))
    start_range0 = 0.01
    my_algo = pygmo.compass_search(max_fevals=1000, start_range=start_range0, stop_range=1.0E-5)
    algo = pygmo.algorithm(my_algo)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)
    fits_local[itrial, :] = pop.champion_x

if RANK == 0:
    fits_all = np.zeros((num_trials*NPROCS, 4))
    buffersizes = 4*num_trials*np.ones((NPROCS,), dtype=int)
    displacements = np.zeros((NPROCS,), dtype=int)
    displacements[1:] = np.cumsum(buffersizes[0:-1])
else:
    fits_all = None
    buffersizes = None
    displacements = None

mpi.COMM_WORLD.Gatherv(sendbuf=fits_local, recvbuf=[fits_all, buffersizes, displacements, mpi.DOUBLE], root=0)

if RANK == 0:
    np.savez("ge_mle_misfits.npz", thetas=fits_all)
