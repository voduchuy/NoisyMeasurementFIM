import sys
sys.path.append("..")
import numpy as np
from bursting_gene.common_settings import NUM_CELLS_FISH, NUM_CELLS_FLOW, NUM_SAMPLING_TIMES, computeCombinedFim
from utils.fim_utils import computeFimFunctional, logTransform

# %%
par_log_transform = True

with np.load("results/fsp_solutions.npz", allow_pickle=True) as fsp_sol_file:
    rna_distributions = fsp_sol_file["rna_distributions"]
    rna_sensitivities = fsp_sol_file["rna_sensitivities"]
    t_meas = fsp_sol_file["t_meas"]

with np.load("results/bursting_parameters.npz") as par:
    kon = par["kon"]
    koff = par["koff"]
    alpha = par["alpha"]
    gamma = par["gamma"]

theta = np.array([kon, koff, alpha, gamma])

# %%
fim_single_cell = dict()

for s in [
    "exact",
    "binomial",
    "poisson_noise",
    "poisson_observation",
    "flowcyt",
    "binomial_state_dep",
]:
    with np.load(f"results/fim_{s}.npz") as data:
        fim_single_cell[s] = data["fim"]

if par_log_transform:
    for fim in fim_single_cell.values():
        logTransform(fim, theta)
#%% D-optimal sampling periods for different types of measurements


fim_multi_cells = dict()

for meas_type in fim_single_cell.keys():
    fim_multi_cells[meas_type] = NUM_CELLS_FISH * fim_single_cell[meas_type]
fim_multi_cells["flowcyt"] = NUM_CELLS_FLOW * fim_single_cell["flowcyt"]

dt_min = 1
dt_max = int(np.floor(t_meas[-1] / NUM_SAMPLING_TIMES))
dt_array = np.linspace(dt_min, dt_max, dt_max - dt_min + 1, dtype=int)

fim_multi_cells_times = dict()
det_fim_multi_cells_times = dict()

for meas_type in fim_multi_cells.keys():
    combined_fim = np.zeros((len(dt_array), 4, 4))
    det_comb_fim = np.zeros((len(dt_array),))

    for i in range(0, len(dt_array)):
        combined_fim[i, :, :] = computeCombinedFim(
            fim_multi_cells[meas_type], dt_array[i], NUM_SAMPLING_TIMES
        )
        det_comb_fim[i] = computeFimFunctional(combined_fim[i, :, :], "d")

    fim_multi_cells_times[meas_type] = combined_fim
    det_fim_multi_cells_times[meas_type] = det_comb_fim
#%%
opt_rates = dict()
for meas in fim_multi_cells_times.keys():
    opt_rates[meas] = np.argmax(det_fim_multi_cells_times[meas])+1
    print(
        f"Optimal sampling period for {meas} is {opt_rates[meas]} "
        f"min with D-opt={det_fim_multi_cells_times[meas][opt_rates[meas]]}."
    )

np.savez(
    "results/opt_sampling_periods.npz",
    opt_rates=opt_rates,
    fim_multi_cells_times=fim_multi_cells_times,
    det_fim_multi_cells_times=det_fim_multi_cells_times,
)
