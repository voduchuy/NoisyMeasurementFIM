import numpy as np
import mpi4py.MPI as mpi
from distortion_models import (
    FlowCytometryModel
)
from numpy.random import SeedSequence, default_rng

ss = SeedSequence(12345)

comm = mpi.COMM_WORLD
cpuid = comm.Get_rank()
comm_size = comm.Get_size()

s = ss.spawn(comm_size)[cpuid]
rng = default_rng(s)
#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
    rna_distributions = f["rna_distributions"]
    rna_sensitivities = f["rna_sensitivities"]
    t_meas = f["t_meas"]
#%% FIMs for continuous-valued flow cytometry measurements
n_iterations = 10
n_particles = 1000000

n_par_local = n_particles // comm_size + (cpuid < (n_particles % comm_size))
flowcyt = FlowCytometryModel()
fim_estimates = []
tmp = np.zeros((1,))
buf = np.zeros((1,))
for imc in range(n_iterations):
    print(f"Monte Carlo iteration {imc}")
    fim = np.zeros((len(t_meas), 4, 4))
    for itime in range(len(t_meas)):
        p = rna_distributions[itime]
        p = np.abs(p)
        p /= np.sum(p)
        xrange = np.arange(len(p))
        xsamples = rng.choice(len(p), p=p, size=n_par_local)
        ysamples = flowcyt.sampleObservations(xsamples, rng=rng)
        yrange, ycounts = np.unique(ysamples, return_counts=True)
        C = flowcyt.getDenseMatrix(xrange, yrange)

        for ip in range(0,4):
            for jp in range(0, 4):
                a = rna_sensitivities[itime][ip]
                b = rna_sensitivities[itime][jp]
                tmp[0] = np.sum(((C@a)/(C@p))*((C@b)/(C@p))*ycounts)

                comm.Allreduce(sendbuf=[tmp, 1, mpi.DOUBLE],
                               recvbuf=[buf, 1, mpi.DOUBLE],
                               op=mpi.SUM)
                fim[itime, ip, jp] = buf[0] / n_particles

    fim_estimates.append(fim)

if cpuid == 0:
    np.savez(f"results/fim_flowcyt.npz", fim=fim_estimates)

