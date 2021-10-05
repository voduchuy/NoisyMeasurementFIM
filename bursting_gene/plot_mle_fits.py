import numpy as np
import matplotlib.pyplot as plt

with np.load("results/ge_mle_misfits.npz") as _:
    fits = _['thetas']

plt.figure()
plt.scatter(fits[-2], fits[-1])
