from distortion_models import DoubleCell
import numpy as np
import matplotlib.pyplot as plt
from common_settings import NUM_CELLS_FISH, NUM_SAMPLING_TIMES, computeSingleObservationFim, computeCombinedFim

#%%
with np.load("results/fsp_solutions.npz", allow_pickle=True) as file:
    rna_distributions = file["rna_distributions"]
    rna_sensitivities = file["rna_sensitivities"]
    t_meas = file["t_meas"]

dt_min = 1
dt_max = int(np.floor(t_meas[-1] / NUM_SAMPLING_TIMES))
dt_array = np.linspace(dt_min, dt_max, dt_max - dt_min + 1, dtype=int)
#%%
error_rates = [0.0] + [0.1, 0.5, 0.8, 0.95, 0.99, 1.0]#np.linspace(0.1, 1.0, 20).tolist()
outputs = {}
for rate in error_rates:
    distortion = DoubleCell(rate)
    distorted_distributions = [
        distortion.transformDistribution(pvec) for pvec in rna_distributions
    ]
    distorted_sensitivities = [
        [distortion.transformSensitivity(rna_distributions[i], svec) for svec in ss]
        for i, ss in enumerate(rna_sensitivities)
    ]
    fims = computeSingleObservationFim(
        distributions=distorted_distributions, sensitivities=distorted_sensitivities
    )
    combined_fims = np.zeros((len(dt_array), 4, 4))
    for i in range(0, len(dt_array)):
        combined_fims[i, :, :] = computeCombinedFim( NUM_CELLS_FISH*fims, dt_array[i], NUM_SAMPLING_TIMES)
    combined_fims_dets = np.array([np.linalg.det(f) for f in combined_fims])
    outputs[rate] = \
    {
        'distorted_distributions': distorted_distributions,
        'distorted_sensitivities': distorted_sensitivities,
        'fims_single_obs': fims,
        'fims_experiment': combined_fims,
        'fims_dets_experiment': combined_fims_dets
    }

#%%
idx = 50
plt.plot(rna_distributions[idx], color="darkgreen")
plt.plot(distorted_distributions[idx], color="red")
plt.show()

#%%
fig = plt.figure(figsize=(4, 4), dpi=300, tight_layout=True)
ax = fig.add_subplot()
for rate in error_rates[1:]:
    ax.plot(dt_array, outputs[rate]['fims_dets_experiment'], label=rate)
ax.plot(dt_array, outputs[0.0]['fims_dets_experiment'], color='k', ls=':', lw=3, label="Exact (0.0)")

ax.set_xlabel("Sampling period (minute)")
ax.set_ylabel("Determinant of FIM")
# ax.set_yscale("log")
legend = fig.legend(ncol=2, bbox_to_anchor=(0, 1, 1, 1), loc="lower left")
legend.set_title("Probability of doublet")
fig.savefig("figs/doublet.png", bbox_inches="tight")
plt.show()
