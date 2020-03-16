import numpy as np
import mpi4py.MPI as mpi
from scipy.stats import norm
import matplotlib.pyplot as plt

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()
np.random.seed(rank)

with np.load('flowcyt_pars.npz') as f:
    kappa = f['kappa']
    sigma_probe = f['sigma_probe']
    mu_bg = f['mu_bg']
    sigma_bg = f['sigma_bg']

with np.load('fsp_solutions.npz', allow_pickle=True) as data:
    rna_distributions = data['rna_distributions']
    rna_sensitivities = data['rna_sensitivities']
nt = len(rna_distributions)

# Monte Carlo estimate of the Fisher Information matrices, we assume that background noise could be accurately quantified independent of the experiment
num_cells_flowcyt_mc = 100000
num_reps = 1

dopt = []

for irep in range(0, num_reps):
    fim_flowcyt = np.zeros((nt, 4, 4))
    mc_size_local = num_cells_flowcyt_mc // num_procs + num_cells_flowcyt_mc % num_procs

    intensity_samples = np.zeros((nt, mc_size_local))

    for itime in range(0, nt):
        xmax = len(rna_distributions[itime]) - 1
        nrna_samples = np.random.choice(xmax + 1, size=(mc_size_local,),
                                        p= np.abs(rna_distributions[itime]) / np.sum(rna_distributions[itime]))
        bg_noise = np.random.normal(loc=mu_bg, scale=sigma_bg, size=(mc_size_local,))
        probe_noise = np.random.normal(loc=0, scale=nrna_samples*sigma_probe, size=(mc_size_local,))
        intensity_samples[itime, :] = kappa * nrna_samples + bg_noise + probe_noise

    def Cmat_flowcyt(itime, p):
        y = np.zeros((mc_size_local,))
        for i in range(0, len(p)):
            y += p[i] * norm.pdf(intensity_samples[itime, :] , loc=mu_bg + kappa * i, scale=sigma_bg + sigma_probe*i)
        return y

    for itime in range(0, nt):
        M = np.zeros((4, 4))
        p = Cmat_flowcyt(itime, rna_distributions[itime])
        si = []
        for ip in range(0, 4):
            si.append(Cmat_flowcyt(itime, rna_sensitivities[itime][ip]))
        for ip in range(0,4):
            for jp in range(0, ip + 1):
                M[ip, jp] = np.sum(si[ip] * si[jp] / np.maximum(p, 1.0e-14))
        for ip in range(0, 4):
            for jp in range(ip + 1, 4):
                M[ip, jp] = M[jp, ip]

        M = comm.allreduce(M)
        fim_flowcyt[itime, :, :] = M / num_cells_flowcyt_mc

    dopt_here = np.zeros((nt,))
    for itime in range(0, nt):
        dopt_here[itime] = np.linalg.det(fim_flowcyt[itime, : , :])
    dopt.append(dopt_here)

    if rank == 0:
        np.savez(f'fim_flowcyt_mc_{irep}.npz', fim_flowcyt=fim_flowcyt)

if rank == 0:
    fig, ax = plt.subplots(1,1)
    for i in range(0, num_reps):
        ax.plot(dopt[i])
    fig.savefig('dopt_estimated.pdf')