# OPTIMIZE PARTIAL OBSERVATIONS
import sys
sys.path.append('..')
import numpy as np
from mapk_model import YeastModel
from utils.fim_utils import computeCriteria, logTransform
#%%
model = YeastModel()
theta = np.array([model.k_01,
                  model.k_10a,
                  model.k_10b,
                  model.k_12,
                  model.k_21,
                  model.k_23,
                  model.k_32,
                  model.alpha_0,
                  model.alpha_1,
                  model.alpha_2,
                  model.alpha_3,
                  model.k_trans,
                  model.gamma_nuc,
                  model.gamma_cyt])
fims = {}
with np.load('results/fim_exact.npz') as _:
    fims['full'] = _["fim"]
with np.load('results/fim_marginals.npz') as _:
    fims['nuc'] = _['nuc_only']
    fims['cyt'] = _['cyt_only']
with np.load('results/fim_total_rna.npz') as _:
    fims['total'] = _['fim']
with np.load('results/fim_joint_rna.npz') as _:
    fims['joint'] = _['fim']
for key in fims.keys():
    logTransform(fims[key], theta)
#%%
def find_multiple_time_fim(dt: int, fims: np.ndarray)->np.ndarray:
    """
    Compute the FIM associated with an experiment that collect cells over 2 equi-spaced time points from the FIMs associated with
    single time points.
    """
    idxs = [k*dt for k in range(0, 5)]
    return np.sum(fims[idxs], axis=0)
#%%
n_cells = 1000
max_dt = 15

fim_dets = np.zeros((n_cells+1, max_dt))

for dt in range(1, max_dt+1):
    for nx in range(0, n_cells+1):
        ny = n_cells - nx
        fims_combined = nx*fims['total'] + ny*fims['nuc']
        f = find_multiple_time_fim(dt, fims_combined)
        fim_dets[nx, dt-1] = np.linalg.det(f)
#%%
nx_opt, dt_opt = np.unravel_index(np.argmax(fim_dets), fim_dets.shape)
fim_opt = find_multiple_time_fim(dt_opt, nx_opt * fims['total'] + (n_cells - nx_opt) * fims['nuc'])
#%%
np.savez('results/combined_fim_dopt.npz', fim_dets=fim_dets, fim_opt=fim_opt, nx_opt=nx_opt, dt_opt=dt_opt)
#%%
import matplotlib.pyplot as plt

plt.contourf(fim_dets)
plt.show()

