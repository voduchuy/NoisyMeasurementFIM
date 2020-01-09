import numpy as np
import mpi4py.MPI as MPI
from chebpy import chebfun
import chebpy
from scipy.stats import logistic
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

kappa = 220
sigma_probe = 390
mu_bg = 200
sigma_bg = 500

with np.load('fsp_solutions.npz', allow_pickle=True) as fsp_sol_file:
    rna_distributions = fsp_sol_file['rna_distributions']
    rna_sensitivities = fsp_sol_file['rna_sensitivities']
    t_meas = fsp_sol_file['t_meas']


# %% Compute the density function of fluorescent intensity and its sensitivities

# This function computes the conditional probability density of the intensity given the number of mRNA molecules
@jit
def intensitycondprob(y, x):
    return np.exp(-((y - x * kappa - mu_bg) ** 2.0) / (2.0 * (x * x * sigma_probe ** 2.0 + sigma_bg ** 2.0))) / \
           np.sqrt(2 * np.pi * (x * x * sigma_probe ** 2.0 + sigma_bg ** 2.0))


# Function to evaluate the pointwise probability density of the intensity
@jit
def intensitypointwise(y, p_x):
    xrange = np.arange(0, len(p_x))
    fval = np.zeros((len(y),))
    for i in range(0, len(y)):
        fval[i] = np.dot(intensitycondprob(y[i], xrange), p_x)
    return fval


flowcyt_intensity_prob = []
flowcyt_intensity_sens = []
for itime in range(0, len(t_meas)):
    xmax = len(rna_distributions[itime]) - 1
    yrange = [mu_bg - 4 * sigma_bg, kappa * xmax + mu_bg + 4 * sigma_bg]
    flowcyt_intensity_prob.append(chebfun(lambda y: intensitypointwise(y, rna_distributions[itime]), yrange))
    stmp = []
    for ip in range(0, 4):
        stmp.append(chebfun(lambda y: intensitypointwise(y, rna_sensitivities[itime][ip]), yrange))
    flowcyt_intensity_sens.append(stmp)


# %%
def computepZfrompY(pY, a, b):
    pZ = np.zeros((2,))
    D = pY.domain
    logisfun = chebfun(lambda x: logistic.cdf(a * (x - b)), D)
    pZ[1] = (logisfun * pY).sum()
    pZ[0] = 1 - pZ[1]
    return pZ


def computeFIMZfromY(sY, pY, a, b):
    pZ = computepZfrompY(pY, a, b)
    npar = len(sY)
    SZ = []
    for j in range(0, npar):
        sZ = computepZfrompY(sY[j], a, b)
        SZ.append(sZ)

    FIMZ = np.zeros((npar, npar))
    for i in range(0, npar):
        for j in range(i, npar):
            FIMZ[i, j] = np.sum(SZ[i] * SZ[j] / pZ)
        for j in range(0, i):
            FIMZ[i, j] = FIMZ[j, i]
    return FIMZ


# %%
ABOUNDS = [1.0E-4, 1.0E3]
BBOUNDS = kappa * [1, 100]
NA = 1000
NB = 100
DT = 300
NCELLS = 1000


def computedetF(a, b):
    t_idx = DT * np.array([1, 2, 3, 4], dtype=int)
    fim = NCELLS * computeFIMZfromY(flowcyt_intensity_sens[t_idx[0]], flowcyt_intensity_prob[t_idx[0]], a, b)
    for i in range(1, len(t_idx)):
        fim += NCELLS * computeFIMZfromY(flowcyt_intensity_sens[t_idx[i]], flowcyt_intensity_prob[t_idx[i]], a, b)
    return np.linalg.det(fim)


NPROCS = MPI.COMM_WORLD.Get_size()
PROCID = MPI.COMM_WORLD.Get_rank()

nblocal = np.empty((1,), dtype=np.intc)
nblocal[0] = NA // NPROCS + int(PROCID < NA % NPROCS)

ibstart = np.zeros((NPROCS + 1,), dtype=np.intc)
nballoc = np.zeros((NPROCS,), dtype=np.intc)
MPI.COMM_WORLD.Allgather(sendbuf=(nblocal, 1, MPI.INT), recvbuf=(nballoc, 1, MPI.INT))
ibstart[1:] = np.cumsum(nballoc)

detFlocal = np.zeros((nblocal[0], NA), dtype=float)
log10abounds = np.log10(ABOUNDS)
log10bbounds = np.log10(BBOUNDS)
for ib in range(0, nblocal[0]):
    for ia in range(0, NA):
        a = np.power(10.0, log10abounds[0] + ia * (log10abounds[1] - log10abounds[0]) / NA)
        b = np.power(10.0, log10bbounds[0] + (ib + ibstart[PROCID]) * (log10bbounds[1] - log10bbounds[0]) / NB)
        detFlocal[ib, ia] = computedetF(a, b)
        print(f'Processor {PROCID} a = {a:.2e} b = {b:.2e} log10(detF) = {np.log10(detFlocal[ib, ia]):.2e} \n')

detF = np.zeros((NB, NA), dtype=float)
recvcounts = nballoc * NA
displs = ibstart[:-1] * NA

MPI.COMM_WORLD.Gatherv(sendbuf=(detFlocal, nblocal[0] * NA, MPI.DOUBLE),
                       recvbuf=(detF, recvcounts, displs, MPI.DOUBLE), root=0)

if PROCID == 0:
    np.savez('detFbinning.npz', detF = detF, abounds=ABOUNDS, bbounds=BBOUNDS)

    detF = np.log10(detF)
    fig, ax = plt.subplots(1,1)
    fig.set_tight_layout(True)
    acoos = np.logspace(log10abounds[0], log10abounds[1], NA)
    bcoos = np.logspace(log10bbounds[0], log10bbounds[1], NB)
    ax.contourf(acoos, bcoos, detF)
    fig.savefig('binning_info_contours.pdf', bbox_inches='tight')



