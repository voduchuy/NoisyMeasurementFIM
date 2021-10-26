# OPTIMIZE PARTIAL OBSERVATIONS
import numpy as np
#%%
fims = {}
with np.load('results/fim_exact.npz') as _:
    fims['full'] = _["fim"]
with np.load('results/fim_marginals.npz') as _:
    fims['partial_0'] = _['fim0']
    fims['partial_1'] = _['fim1']
#%%
def find_multiple_time_fim(dt: int, fims: np.ndarray)->np.ndarray:
    idxs = [k*dt for k in range(1, 3)]
    return np.sum(fims[idxs], axis=0)
#%%
n_cells = 1000
max_dt = 200

fim_dets = np.zeros((n_cells+1, max_dt))

for dt in range(1, max_dt+1):
    for nx in range(0, n_cells+1):
        ny = n_cells - nx
        fims_combined = nx*fims['partial_0'] + ny*fims['partial_1']
        f = find_multiple_time_fim(dt, fims_combined)
        fim_dets[nx, dt-1] = np.linalg.det(f)
#%%
nx_opt, dt_opt = np.unravel_index(np.argmax(fim_dets), fim_dets.shape)
fim_opt = find_multiple_time_fim(dt_opt, nx_opt * fims['partial_0'] + (n_cells - nx_opt) * fims['partial_1'])
#%%
np.savez('results/combined_fim_dopt.npz', fim_dets=fim_dets, fim_opt=fim_opt, nx_opt=nx_opt, dt_opt=dt_opt)
#%%
import matplotlib.pyplot as plt

plt.contourf(fim_dets)
plt.show()

