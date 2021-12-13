# OPTIMIZE PARTIAL OBSERVATIONS
import sys
sys.path.append('..')
import numpy as np
from toggle_model import ToggleSwitchModel
from utils.fim_utils import compute_fim_functional, log_transform
#%%
model = ToggleSwitchModel()
theta = np.array([model.bx,
                model.by,
                model.kx,
                model.ky,
                model.ayx,
                model.axy,
                model.nyx,
                model.nxy,
                model.gammax,
                model.gammay])
fims = {}
with np.load('results/fim_exact.npz') as _:
    fims['full'] = _["fim"]
with np.load('results/fim_marginals.npz') as _:
    fims['partial_0'] = _['fim0']
    fims['partial_1'] = _['fim1']
for key in fims.keys():
    log_transform(fims[key], theta)
#%%
n_cells = 1000
max_dt = 200

fim_dets = np.zeros((n_cells+1, max_dt))

for dt in range(1, max_dt+1):
    for nx in range(0, n_cells+1):
        ny = n_cells - nx
        fims_combined = nx*fims['partial_0'] + ny*fims['partial_1']
        fim_dets[nx, dt-1] = np.linalg.det(fims_combined[dt])
#%%
nx_opt, dt_opt = np.unravel_index(np.argmax(fim_dets), fim_dets.shape)
fim_opt = nx_opt * fims['partial_0'][dt_opt] + (n_cells - nx_opt) * fims['partial_1'][dt_opt]
#%%
np.savez('results/combined_fim_dopt.npz', fim_dets=fim_dets, fim_opt=fim_opt, nx_opt=nx_opt, dt_opt=dt_opt)
#%%
import matplotlib.pyplot as plt

plt.contourf(fim_dets)
plt.show()

