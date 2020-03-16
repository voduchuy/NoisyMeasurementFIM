#%% This script reads the mRNA distributions and sensitivities computed by the FSP, and outputs an estimation of the FIM for the mean measurement
import mpi4py.MPI as mpi
import numpy as np
#%% Common parameters
pseudodata_size = 2000 # Number of cells in each experimental dataset
num_reps = 10000
t_meas = [170]
#%% Read the FSP solution
with np.load('fsp_solutions.npz', allow_pickle=True) as file:
    p_rna = file['rna_distributions']
    s_rna = file['rna_sensitivities']

#%% Generate pseudo-samples and compute their sample mean estimates
pseudosample_means = np.zeros((num_reps,))
for i in range(0, num_reps):
    pvector = p_rna[t_meas[0]]
    pvector[pvector < 0.0] = 0.0
    pvector = pvector/np.sum(pvector)
    xsamples = np.random.choice(len(pvector), size=pseudodata_size, p=pvector)
    pseudosample_means[i] = np.mean(xsamples)
#%% The exact mean
    exact_mean = np.dot(np.arange(0, len(pvector)), pvector)
    exact_var = np.dot(np.arange(0, len(pvector))*np.arange(0, len(pvector)), pvector) - exact_mean*exact_mean
#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1,1)
sns.distplot(pseudosample_means, ax=ax)
ax.axvline(exact_mean)
ax.axvline(exact_mean - 3*np.sqrt(exact_var/pseudodata_size))
ax.axvline(exact_mean + 3*np.sqrt(exact_var/pseudodata_size))
plt.show()