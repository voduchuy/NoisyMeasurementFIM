import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from utils.fim_utils import logTransform


with np.load("results/fims.npz") as _:
    taus = _["taus"]
    fims = _["fims"]
    bnodes = _["bnodes"]
    dnodes = _["dnodes"]

bb, dd = np.meshgrid(bnodes, dnodes)
for i in range(len(bnodes)):
    for j in range(len(dnodes)):
        logTransform(fims=fims[i, j, ...], theta=[bb[i, j], dd[i, j]])

#%%
cov_ests = np.zeros((len(bnodes), len(dnodes), len(taus), 2, 2))
for i in range(len(bnodes)):
    for j in range(len(dnodes)):
        for it in range(len(taus)):
            cov_ests[i, j, it, :, :] = np.linalg.inv(1000.0*np.sum(fims[i, j, it, :, :, :], axis=0))

