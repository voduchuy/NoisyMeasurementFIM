# Visualze the model manifold of bursting gene under experimental distortion using the intensive embedding proposed by Jim Sethna's group
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import sklearn.manifold as manifold

k_off = 0.015
k_on = 0.05
k_r = 5
gamma = 0.05

k_off_bounds = [0.01, 0.1]
k_on_bounds = [0.01, 0.1]
k_r_bounds = [0.1, 10]
gamma_bounds = [0.01, 0.1]

t_final = 60

SM = [[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]]
X0 = [[1, 0, 0]]
P0 = [1.0]
S0 = [0.0]

def prop_factory(theta):
    [k_off, k_on, k_r, gamma] = theta[:]
    def prop_x(reaction, X, out):
        if reaction == 0:
            out[:] = k_on*X[:, 0]
            return None
        if reaction == 1:
            out[:] = k_off*X[:, 1]
            return None
        if reaction == 2:
            out[:] = k_r*X[:, 1]
            return None
        if reaction == 3:
            out[:] = gamma*X[:, 2]
            return None
    return prop_x


n_rna_max = 200

def solve_model(theta):
    init_bounds = np.array([1, 1, 20])
    solver = FspSolverMultiSinks(mpi.COMM_SELF)
    prop_x = prop_factory(theta)
    solver.SetModel(np.array(SM), None, prop_x)
    solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
    solver.SetInitialDist(np.array(X0), np.array(P0))
    solution = solver.Solve(t_final, 1.0e-4)

    rna_dist = solution.Marginal(2)
    rna_dist = rna_dist[0:n_rna_max+1]
    return rna_dist

#%%
num_samples = 100
thetas = np.random.uniform(low=[k_off_bounds[0], k_on_bounds[0], k_r_bounds[0], gamma_bounds[0]],
                           high=[k_off_bounds[1], k_on_bounds[1], k_r_bounds[1], gamma_bounds[1]],
                           size=(num_samples, 4))
#%%
embedded_dists = np.zeros((num_samples, n_rna_max+1), dtype=float)

for i in range(0, thetas.shape[0]):
    theta = thetas[i , :]
    sqrtp = np.sqrt(np.abs(solve_model(theta)))
    n = np.minimum(len(sqrtp), n_rna_max+1)
    embedded_dists[i, :n] = sqrtp[0:n]
    print(i)

#%% Compute the distance metric based on the intensive embedding
dist_mat = np.zeros((num_samples, num_samples))
for i in range(0, num_samples):
    for j in range(i+1,num_samples):
        dist_mat[i, j] = -8.0*np.log(np.dot(embedded_dists[i,:], embedded_dists[j,:]))
    for j in range(0, i):
        dist_mat[i, j] = dist_mat[j, i]

#%% Compute the low-dim visualization
map = manifold.Isomap(metric='precomputed', n_components=2)
X_transformed = map.fit_transform(dist_mat)

import matplotlib.pyplot as plt
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()

