from scipy.optimize import minimize, Bounds
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.sensitivity import SensFspSolverMultiSinks

# %% Number of MLE samples to generate
# num_mle = 4
# %% Experiment design
n_cells = 1000
t_meas = np.linspace(1, 10, 100)
idx = 12
# %%
comm = mpi.COMM_WORLD
comm_size = comm.size
rank = comm.rank
np.random.seed(rank)

# Number of MLE samples this processor has to generate
my_num_mle = 250

# %% Define model structure
stoich_mat = np.array([[-1, 1, 0],
                       [1, -1, 0],
                       [0, 0, 1],
                       [0, 0, -1]])
x0 = np.array([[1, 0, 0]])
p0 = np.array([1.0])
s0 = np.array([0.0])
constr_init = np.array([1, 1, 100])

k_off = 0.5
k_on = 0.1
k_r = 50.0
gamma = 1.0
theta_true = np.array([k_off, k_on, k_r, gamma])
varying_par = np.array([0, 1, 2, 3], dtype=int)
num_uncertain_parameters = varying_par.size


def propensity(reaction, x, out):
    if reaction is 0:
        out[:] = x[:, 0]
    if reaction is 1:
        out[:] = x[:, 1]
    if reaction is 2:
        out[:] = x[:, 1]
    if reaction is 3:
        out[:] = x[:, 2]


def t_fun_factory(theta):
    def t_fun(t, out):
        out[:] = theta[:]

    return t_fun


def dt_fun_factory(theta):
    dt_list = []

    def d_t_fun(i):
        def d_t_(t, out):
            out[i] = 1.0

        return d_t_

    for i in range(0, 4):
        dt_list.append(d_t_fun(i))
    return dt_list


# %% Simulate a data set

def simulate_data():
    ssa = SSASolver()
    ssa.SetModel(stoich_mat, t_fun_factory(theta_true), propensity)
    observations = ssa.Solve(t_meas[idx], x0, n_cells)
    return SmFishSnapshot(observations[:, 2])


# %%
def obj_func(log_theta, dat):
    theta = np.copy(theta_true)
    theta[varying_par] = np.exp(log_theta)
    # print(theta)
    t_fun = t_fun_factory(theta)
    solver = FspSolverMultiSinks(mpi.COMM_SELF)
    solver.SetModel(stoich_mat, t_fun, propensity)
    solver.SetFspShape(None, constr_init)
    solver.SetOdeSolver("Krylov")
    solver.SetInitialDist(x0, p0)
    solver.SetUp()
    try:
        solution = solver.Solve(t_meas[idx], 1.0e-6)
        ll = -1.0 * dat.LogLikelihood(solution, np.array([2]))
    except:
        ll = np.Infinity
    return ll


# %%
def mle_fit():
    log_theta0 = np.log(np.copy(theta_true[varying_par]))
    data = simulate_data()
    res = minimize(obj_func, np.copy(log_theta0), args=data, method='Nelder-Mead', options={'disp': True})
    return res['x']


my_mle = np.zeros((my_num_mle, num_uncertain_parameters))
for i in range(0, my_num_mle):
    my_mle[i, :] = mle_fit()

all_mle = np.zeros((comm_size * my_num_mle, num_uncertain_parameters))

send_count = num_uncertain_parameters * my_num_mle * np.ones((comm_size,), dtype=int)

displacements = np.zeros((comm_size,), dtype=int)
if comm_size > 1:
    displacements[1:] = np.cumsum(send_count[0:comm_size - 1])
comm.Gatherv(my_mle, [all_mle, tuple(send_count), tuple(displacements), mpi.DOUBLE])

if rank == 0:
    print(all_mle)
    np.save('cge_fits', arr=all_mle)