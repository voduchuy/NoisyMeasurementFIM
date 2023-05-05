# This script computes the determinant of the FIM for the bursting gene measurements
# associated with different sampling periods and binning widths, assuming uniform binning.
import sys

sys.path.append("..")
import numpy as np
from utils.fim_utils import computeFimFunctional, logTransform
from common_settings import NUM_SAMPLING_TIMES, computeCombinedFim
from distortion_models import UniformBinning
from common_settings import computeFimEntry, computeSingleObservationFim

#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
    rna_distributions = f["rna_distributions"]
    rna_sensitivities = f["rna_sensitivities"]
    t_meas = f["t_meas"]

with np.load("results/bursting_parameters.npz") as par:
    kon = par["kon"]
    koff = par["koff"]
    alpha = par["alpha"]
    gamma = par["gamma"]

theta = np.array([kon, koff, alpha, gamma])
#%%
dtToComputeAt = 30
step = 1
binWidths = np.arange(0, 60, step=step)
binWidths[0] = 1
fimDets = np.zeros((binWidths.shape[0],))
#%%
nMax = 300
nCells = 1000
xrange = np.arange(nMax)
yrange = xrange
fims = {}
for i, width in enumerate(binWidths):
    C = UniformBinning(width).getDenseMatrix(xrange, yrange)
    fimSingles = computeSingleObservationFim(
        distributions=rna_distributions,
        sensitivities=rna_sensitivities,
        distortionMatrix=C,
    )
    logTransform(fimSingles, theta)
    fimDets[i] = np.linalg.det(
        nCells
        * computeCombinedFim(
            fim_array=fimSingles, dt=dtToComputeAt, num_times=NUM_SAMPLING_TIMES
        )
    )
    fims[width] = nCells * computeCombinedFim(
        fim_array=fimSingles, dt=dtToComputeAt, num_times=NUM_SAMPLING_TIMES
    )
#%%
np.savez(
    "results/binning_sweep.npz",
    binWidths=binWidths,
    dtToComputeAt=dtToComputeAt,
    fimDets=fimDets,
    fims=fims,
)
