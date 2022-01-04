# This script computes the determinant of the FIM for the bursting gene measurements
# associated with different sampling periods and detection rate, assuming a binomial distortion operator.
import sys
sys.path.append("..")
import numpy as np
from utils.fim_utils import computeFimFunctional, logTransform
from common_settings import NUM_SAMPLING_TIMES, computeCombinedFim
from distortion_models import BinomialDistortionModel
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
nRate = 100
nSampling = dtMax
detectionRateGrid = np.linspace(0, 1, nRate)
samplingPeriodGrid = np.arange(nSampling)
fimDets = np.zeros((nRate, nSampling))
dtOpts = np.zeros((nRate,))
#%%
nMax = 300
nCells = 1000
xrange = np.arange(nMax)
yrange = xrange
for iRate, detectionRate in enumerate(detectionRateGrid):
    C = BinomialDistortionModel(detectionRate=detectionRate).getDenseMatrix(xrange, yrange)
    fimSingles = computeSingleObservationFim(distributions=rna_distributions, sensitivities=rna_sensitivities, distortionMatrix=C)
    logTransform(fimSingles, theta)
    for dt in range(dtMax):
        fimDets[iRate, dt] = np.linalg.det(computeCombinedFim(fim_array=nCells*fimSingles, dt=dt, num_times=NUM_SAMPLING_TIMES))
        dtOpts[iRate] = np.argmax(fimDets[iRate,:])
#%%
np.savez("results/binomial_sweep.npz", dtOpts=dtOpts, detectionRateGrid=detectionRateGrid, dtMax=dtMax, fimDets=fimDets)

