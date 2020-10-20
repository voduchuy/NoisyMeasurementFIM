import numpy as np
import mpi4py.MPI as mpi

COMM = mpi.COMM_WORLD
RANK = mpi.COMM_WORLD.Get_rank()
NCPU = mpi.COMM_WORLD.Get_size()

par_log_transform = True # Transform the FIM to log-parameter space

with np.load('results/bursting_parameters.npz') as par:
    k_on = par['kon']
    k_off = par['koff']
    k_r = par['kr']
    gamma = par['gamma']

theta = np.array([k_on, k_off, k_r, gamma])

with np.load('results/fsp_solutions.npz', allow_pickle=True) as fsp_sol_file:
    rna_distributions = fsp_sol_file['rna_distributions']
    rna_sensitivities = fsp_sol_file['rna_sensitivities']
    t_meas = fsp_sol_file['t_meas']

fim_single_cell = dict()

with np.load('results/fim_exact.npz') as data:
    fim_single_cell['exact'] = data['fim_exact']

with np.load('results/fim_binom.npz') as data:
    fim_single_cell['binom'] = data['fim_binom']

with np.load('results/fim_lowres.npz') as data:
    fim_single_cell['lowres'] = data['fim_lowres']

with np.load('results/fim_composite.npz') as data:
    fim_single_cell['composite'] = data['fim_composite']

with np.load('results/fim_flowcyt.npz') as data:
    fim_single_cell['flowcyt'] = data['fim_flowcyt']

if par_log_transform:
    for fim in fim_single_cell.values():
        for it in range(0, len(t_meas)):
            for i in range(0, 4):
                for j in range(0, 4):
                    fim[it, i, j] *= theta[i] * theta[j] * np.log(10) * np.log(10)

#%%
def compute_multitime_fim(fim_array: np.ndarray,
                          dt: int,
                          num_times: int):
    t_idx = dt*np.linspace(1, num_times, num_times, dtype=int)
    fim = fim_array[t_idx[0], :, :]
    for i in range(1, len(t_idx)):
        fim += fim_array[t_idx[i], :, :]
    return fim
#%%
p_grid = np.linspace(0, 1, 100)

nsmfish_max = 1000
nflowcyt_max = 10000

num_times = 5
dt_min = 1
dt_max = int(np.floor(t_meas[-1]/num_times))
dt_array = np.linspace(dt_min, dt_max, dt_max - dt_min +1, dtype=int)

# Figure out the distribution of ps to all cpus
num_ps_local = np.array([len(p_grid) // NCPU + (RANK < len(p_grid) % NCPU)], dtype=int)
ps_distribution = np.zeros((NCPU,), dtype=int)
COMM.Allgather(sendbuf=(num_ps_local, 1, mpi.INT), recvbuf=(ps_distribution, mpi.INT))
ps_displacements = np.zeros((NCPU,), dtype=int)
ps_displacements[1:] = np.cumsum(ps_distribution[0:-1])

# Compute the local multifidelity FIMs, each CPU compute their subset of the mixing parameter grid
fim_mats_local = np.zeros((num_ps_local[0], len(dt_array), 4, 4), dtype=float)
for ip in range(0, num_ps_local[0]):
    fim_mix = np.zeros((len(t_meas), 4, 4))
    for it in range(0, len(t_meas)):
        fim_mix[it, :, :] = p_grid[ip + ps_displacements[RANK]] * nsmfish_max*fim_single_cell['lowres'][it] + (1-p_grid[
            ip+ps_displacements[RANK]]) * nflowcyt_max*fim_single_cell['flowcyt'][it]
    for i in range(0, len(dt_array)):
        fim_mats_local[ip, i, :, :] = compute_multitime_fim(fim_mix, dt_array[i], num_times)

# Send them all to CPU 0 for saving
if RANK == 0:
    fim_mats_glob = np.zeros((len(p_grid), len(dt_array), 4, 4), dtype=float)
    buffer_sizes = len(dt_array)*16*ps_distribution
    buffer_displs = len(dt_array)*16*ps_displacements
else:
    fim_mats_glob = None
    buffer_sizes = None
    buffer_displs = None

COMM.Gatherv(sendbuf=(fim_mats_local, num_ps_local[0]*len(dt_array)*16, mpi.DOUBLE), recvbuf=(fim_mats_glob,
                                                                                             buffer_sizes,
                                                                             buffer_displs, mpi.DOUBLE))

if RANK == 0:
    np.savez('results/fim_multifi.npz', fim_multifi = fim_mats_glob, p_grid = p_grid, dt_array = dt_array)
