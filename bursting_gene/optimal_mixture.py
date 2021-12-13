# This script find an optimal mixture of smFISH experiment and flow cytometry measurement
import sys
sys.path.append("..")
import numpy as np
from utils.fim_utils import compute_fim_functional, log_transform
from common_settings import num_sampling_times, compute_combined_fim

BUDGET_MAX = 1000.0
DT_MEASUREMENT = 30
CRITERIA = "d"
#%%
class MeasurementMethod(object):
    def __init__(self, codename: str="binomial", title: str="", cost: float=1.0):
        self.codename = codename
        self.title = title
        self.fim = None
        self.cost = cost

    def load_fim(self, do_log_transform=True):
        with np.load(f'results/fim_{self.codename}.npz') as _:
            self.fim = _['fim']
        if do_log_transform:
            with np.load('results/bursting_parameters.npz') as par:
                kon = par['kon']
                koff = par['koff']
                alpha = par['alpha']
                gamma = par['gamma']
            theta = np.array([kon, koff, alpha, gamma])
            log_transform(self.fim, theta)
#%%
methods = []
methods.append(MeasurementMethod("binomial", "smFISH with random missing spots", 1.0))
methods.append(MeasurementMethod("binomial_poisson", "smFISH with random missing spots and additive Poisson noise", 0.6))
#%%
for method in methods:
    method.load_fim()
#%%
obj_values = np.zeros(
                (
                    int(BUDGET_MAX//methods[0].cost)+1,
                    int(BUDGET_MAX//methods[1].cost)+1
                )
)
component_fims = [
    compute_combined_fim(method.fim, DT_MEASUREMENT, num_sampling_times)
    for method in methods
]
for n0 in range(obj_values.shape[0]):
    n1_max = int((BUDGET_MAX-n0*methods[0].cost)//methods[1].cost)
    for n1 in range(n1_max+1):
        fim_combined = n0*component_fims[0] + n1*component_fims[1]
        obj_values[n0, n1] = compute_fim_functional(fim_combined, criteria=CRITERIA)
#%%
opt_mixture = np.unravel_index(np.argmax(obj_values), obj_values.shape)
fim_mix_opt = opt_mixture[0]*component_fims[0] + opt_mixture[1]*component_fims[1]

print(f"""
      The optimal mixture is {opt_mixture[0]} {methods[0].title} and
      {opt_mixture[1]} {methods[1].title} with D-opt value {np.max(obj_values): .2e}.
      """)
#%%
np.savez("results/opt_mixture.npz",
         opt_mixture=opt_mixture,
         obj_values=obj_values,
         fim_mix_opt=fim_mix_opt)
