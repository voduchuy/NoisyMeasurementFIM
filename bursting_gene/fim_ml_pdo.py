from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np

# This problem actually can be solved with just one core, but why leave your other cores idle?
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()
#%%
if rank == 0:
    with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
        rna_distributions = f["rna_distributions"]
        rna_sensitivities = f["rna_sensitivities"]
        t_meas = f["t_meas"]
else:
    rna_distributions = None
    rna_sensitivities = None
    t_meas = None

rna_distributions = comm.bcast(rna_distributions)
rna_sensitivities = comm.bcast(rna_sensitivities)
t_meas = comm.bcast(t_meas)

#%%

if rank == 0:
    num_pixels = 100  # number of pixels occupied by a typical cells
    with np.load("results/ml_lowres_C_matrix.npz") as f:
        C_lowres = f["C_lowres"]

    fim_lowres = np.zeros((len(t_meas), 4, 4))
    for itime in range(0, len(t_meas)):
        M = np.zeros((4, 4))
        xmax = len(rna_distributions[itime]) - 1
        p = C_lowres[0 : xmax + 1, 0 : xmax + 1] @ rna_distributions[itime]
        for ip in range(0, 4):
            sip = C_lowres[0 : xmax + 1, 0 : xmax + 1] @ rna_sensitivities[itime][ip]
            for jp in range(0, ip + 1):
                sjp = (
                    C_lowres[0 : xmax + 1, 0 : xmax + 1] @ rna_sensitivities[itime][jp]
                )
                M[ip, jp] = np.sum(sip * sjp / np.maximum(1.0e-16, p))
        for ip in range(0, 4):
            for jp in range(ip + 1, 4):
                M[ip, jp] = M[jp, ip]
        fim_lowres[itime, :, :] = M
    np.savez("results/fim_lowres_ml.npz", fim_lowres=fim_lowres)
