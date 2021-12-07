# %%

par_log_transform = True

# %%

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
from utils.my_plotting_lib import plot_barcodes, plot_conf_ellipse, label_axes
from utils.fim_utils import compute_fim_functional, log_transform

rc('text', usetex=True)

# %%

labels = {
    'exact': f'smFISH exact',
    'binomial': f'smFISH missing spots',
    'flowcyt': f'Flow cytometry',
    'poisson': f'Additive Poisson noise'
}

colors = {
    'exact': f'red',
    'binomial': f'blue',
    'flowcyt': f'darkgreen',
    'poisson': 'magenta'
}

with np.load('results/fsp_solutions.npz', allow_pickle=True) as fsp_sol_file:
    rna_distributions = fsp_sol_file['rna_distributions']
    rna_sensitivities = fsp_sol_file['rna_sensitivities']
    t_meas = fsp_sol_file['t_meas']

with np.load('results/bursting_parameters.npz') as par:
    kon = par['kon']
    koff = par['koff']
    alpha = par['alpha']
    gamma = par['gamma']

theta = np.array([kon, koff, alpha, gamma])

# %%

fim_single_cell = dict()

for s in ["exact", "binomial", "poisson"]:
    with np.load(f'results/fim_{s}.npz') as data:
        fim_single_cell[s] = data['fim']
with np.load(f'results/fim_flowcyt.npz') as _:
    fim_flowcyt_samples = _['fim']
fim_flowcyt = np.mean(fim_flowcyt_samples, axis=0)
fim_single_cell["flowcyt"] = fim_flowcyt

if par_log_transform:
    for fim in fim_single_cell.values():
        log_transform(fim, theta)
#%% D-optimal sampling periods for different types of measurements


ncells_smfish = 1000
ncells_flowcyt = 1000

fim_multi_cells = dict()

for meas_type in fim_single_cell.keys():
    fim_multi_cells[meas_type] = ncells_smfish * fim_single_cell[meas_type]
fim_multi_cells['flowcyt'] = ncells_flowcyt * fim_single_cell['flowcyt']


def compute_combined_fim(fim_array, dt, num_times):
    t_idx = dt * np.linspace(1, num_times, num_times, dtype=int)
    fim = fim_array[t_idx[0], :, :]
    for i in range(1, len(t_idx)):
        fim += fim_array[t_idx[i], :, :]
    return fim


num_times = 5
dt_min = 1
dt_max = int(np.floor(t_meas[-1] / num_times))
dt_array = np.linspace(dt_min, dt_max, dt_max - dt_min + 1, dtype=int)

fim_multi_cells_times = dict()
det_fim_multi_cells_times = dict()

for meas_type in fim_multi_cells.keys():
    combined_fim = np.zeros((len(dt_array), 4, 4))
    det_comb_fim = np.zeros((len(dt_array),))

    for i in range(0, len(dt_array)):
        combined_fim[i, :, :] = compute_combined_fim(fim_multi_cells[meas_type], dt_array[i], num_times)
        det_comb_fim[i] = compute_fim_functional(combined_fim[i, :, :], "d")

    fim_multi_cells_times[meas_type] = combined_fim
    det_fim_multi_cells_times[meas_type] = det_comb_fim
#
opt_rates = dict()
for meas in fim_multi_cells_times.keys():
    opt_rates[meas] = np.argmax(det_fim_multi_cells_times[meas])
    print(f"Optimal sampling period for {meas} is {opt_rates[meas]} "
          f"min with D-opt={det_fim_multi_cells_times[meas][opt_rates[meas]]}.")
