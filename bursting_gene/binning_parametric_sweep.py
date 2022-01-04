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

with np.load('results/bursting_parameters.npz') as par:
    kon = par['kon']
    koff = par['koff']
    alpha = par['alpha']
    gamma = par['gamma']

theta = np.array([kon, koff, alpha, gamma])
#%%
dtMax = int(np.floor(t_meas[-1] / NUM_SAMPLING_TIMES))
step = 1
nSampling = dtMax
binWidths = np.arange(0, 60, step=step)
binWidths[0] = 1
samplingPeriodGrid = np.arange(nSampling)
fimDets = np.zeros((binWidths.shape[0], nSampling))
#%%
nMax = 300
nCells = 1000
xrange = np.arange(nMax)
yrange = xrange
optimalFims = {}
dtOpts = np.zeros((binWidths.shape[0]), dtype=np.intc)
for i, width in enumerate(binWidths):
    C = UniformBinning(width).getDenseMatrix(xrange, yrange)
    fimSingles = computeSingleObservationFim(distributions=rna_distributions, sensitivities=rna_sensitivities, distortionMatrix=C)
    logTransform(fimSingles, theta)
    for dt in range(dtMax):
        fimDets[i, dt] = np.linalg.det(nCells * computeCombinedFim(fim_array=fimSingles, dt=dt, num_times=NUM_SAMPLING_TIMES))
    dtOpts[i] = np.argmax(fimDets[i,:])
    optimalFims[width] = nCells*computeCombinedFim(fim_array=fimSingles, dt=dtOpts[i], num_times=NUM_SAMPLING_TIMES)
#%%
np.savez("results/binning_sweep.npz", dtOpts=dtOpts, binWidths=binWidths, dtMax=dtMax, fimDets=fimDets, optimalFims = optimalFims)

