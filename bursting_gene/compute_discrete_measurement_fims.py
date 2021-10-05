import numpy as np
from distortion_models import (
    BinomialDistortionModel,
    LowResolutionModel,
    FlowCytometryModel,
    RNG
)

#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
    rna_distributions = f["rna_distributions"]
    rna_sensitivities = f["rna_sensitivities"]
    t_meas = f["t_meas"]
#%%
def fisher_metric(a, b, p, C):
    xmax = len(p) - 1
    Cp = C[:, 0 : xmax + 1] @ p
    Ca = C[:, 0 : xmax + 1] @ a
    Cb = C[:, 0 : xmax + 1] @ b
    return np.sum(Ca * Cb / np.maximum(1.0e-16, Cp))


#%%
# FIM for exact smFISH measurements
fim_exact = np.zeros((len(t_meas), 4, 4))
for itime in range(0, len(t_meas)):
    M = np.zeros((4, 4))
    for ip in range(0, 4):
        for jp in range(0, ip + 1):
            M[ip, jp] = np.sum(
                rna_sensitivities[itime][ip]
                * rna_sensitivities[itime][jp]
                / np.maximum(rna_distributions[itime], 1.0e-16)
            )
    for ip in range(0, 4):
        for jp in range(ip + 1, 4):
            M[ip, jp] = M[jp, ip]
    fim_exact[itime, :, :] = M
np.savez("results/fim_exact.npz", fim=fim_exact)
#%% FIM for discrete-valued measurements
discrete_measurements_dict = {
    "binomial": BinomialDistortionModel,
    "lowres": LowResolutionModel
}

n_max = 400

for name, noise_model in discrete_measurements_dict.items():
    xrange = np.arange(n_max)
    yrange = xrange

    C = noise_model().getDenseMatrix(yrange, xrange)

    fim = np.zeros((len(t_meas), 4, 4))
    for itime in range(0, len(t_meas)):
        for ip in range(0, 4):
            for jp in range(0, ip+1):
                fim[itime, ip, jp] = fisher_metric(
                    rna_sensitivities[itime][ip],
                    rna_sensitivities[itime][jp],
                    rna_distributions[itime],
                    C,
                )
        for ip in range(0, 4):
            for jp in range(ip + 1, 4):
                fim[itime, ip, jp] = fim[itime, jp, ip]

    np.savez(f"results/fim_{name}.npz", fim=fim)
    np.savez(f"results/distortion_matrix_{name}.npz", C=C, xrange=xrange, yrange=yrange)


