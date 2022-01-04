import sys 
sys.path.append("..")
import numpy as np
from distortion_models import SmFishDistortion
from utils.fim_utils import computeFimFunctional, logTransform
from common_settings import computeFimEntry, computeSingleObservationFim, NUM_SAMPLING_TIMES, computeCombinedFim
#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
    rna_distributions = f["rna_distributions"]
    rna_sensitivities = f["rna_sensitivities"]
    t_meas = f["t_meas"]
with np.load("results/smfish_probe_distortions.npz", allow_pickle=True) as _:
    probeLevels = _["levels"]
    pdos = _["pdos"][()]
with np.load("results/bursting_parameters.npz") as par:
    kon = par["kon"]
    koff = par["koff"]
    alpha = par["alpha"]
    gamma = par["gamma"]

theta = np.array([kon, koff, alpha, gamma])
#%%
fims = {}
for level in probeLevels:
    fims[level] = computeSingleObservationFim(distributions=rna_distributions, sensitivities=rna_sensitivities, distortionMatrix=pdos[level])
    logTransform(fims[level], theta)

#%%
dtMin = 1
dtMax = int(np.floor(t_meas[-1] / NUM_SAMPLING_TIMES))
dtArray = np.linspace(dtMin, dtMax, dtMax - dtMin + 1, dtype=int)
numCells = 1000
fimMultiCellsTimes = {}
detFimMultiCellsTimes = {}

for level in probeLevels:
    combinedFim = np.zeros((len(dtArray), 4, 4))
    detCombFim = np.zeros((len(dtArray),))
    for i in range(0, len(dtArray)):
        combinedFim[i, :, :] = computeCombinedFim(
            numCells * fims[level], dtArray[i], NUM_SAMPLING_TIMES
        )
        detCombFim[i] = computeFimFunctional(combinedFim[i, :, :], "d")

    fimMultiCellsTimes[level] = combinedFim
    detFimMultiCellsTimes[level] = detCombFim
#%%
opt_rates = dict()
for level in fimMultiCellsTimes.keys():
    opt_rates[level] = np.argmax(detFimMultiCellsTimes[level])
    print(
        f"Optimal sampling period for {level} is {opt_rates[level]} "
        f"min with D-opt={detFimMultiCellsTimes[level][opt_rates[level]]}."
    )
#%%
np.savez("results/fim_probe_binding.npz", levels=probeLevels, fim_single_obs=fims, fim_multi_cells_times=fimMultiCellsTimes, det_fim_multi_cells_times=detFimMultiCellsTimes, opt_dts=opt_rates)