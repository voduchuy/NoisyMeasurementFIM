import sys

sys.path.append("..")
import numpy as np
from bursting_gene.distortion_models import (
    BinomialDistortionModel,
    AdditivePoissonDistortionModel,
    BinomialVaryingDetectionRate,
    PoissonObservationModel
)
from bursting_gene.common_settings import computeFimEntry, computeSingleObservationFim

#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as f:
    rna_distributions = f["rna_distributions"]
    rna_sensitivities = f["rna_sensitivities"]
    t_meas = f["t_meas"]
#%%
# FIM for exact smFISH measurements
fim_exact = computeSingleObservationFim(
    distributions=rna_distributions, sensitivities=rna_sensitivities
)
np.savez("results/fim_exact.npz", fim=fim_exact)
#%% FIM for discrete-valued measurements
discrete_measurements_dict = {
    "binomial": BinomialDistortionModel,
    "binomial_state_dep": BinomialVaryingDetectionRate,
    "poisson_noise": AdditivePoissonDistortionModel,
    "poisson_observation": PoissonObservationModel
}

n_max = 400

for name, noise_model in discrete_measurements_dict.items():
    xrange = np.arange(n_max)
    yrange = xrange

    C = noise_model().getDenseMatrix(yrange, xrange)
    fim = computeSingleObservationFim(
        distributions=rna_distributions,
        sensitivities=rna_sensitivities,
        distortionMatrix=C,
    )
    np.savez(f"results/fim_{name}.npz", fim=fim)
    np.savez(
        f"results/distortion_matrix_{name}.npz", C=C, xrange=xrange, yrange=yrange
    )
