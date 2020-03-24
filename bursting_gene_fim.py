from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np

k_off = 0.015
k_on = 0.05
k_r = 5
gamma = 0.05

t_meas = np.linspace(0, 400, 401)

theta = np.array([k_on, k_off, k_r, gamma])

SM = [[-1, 1, 0],
      [1, -1, 0],
      [0, 0, 1],
      [0, 0, -1]]

X0 = [[1, 0, 0]]
P0 = [1.0]
S0 = [0.0]


def dprop_t_factory(i):
    def dprop_t(t, out):
        out[:] = 0.0
        out[i] = 1.0
    return dprop_t


dprop_t_list = []
for i in range(0, 4):
    dprop_t_list.append(dprop_t_factory(i))
dprop_sparsity = np.eye(4, dtype=np.intc)


def prop_t(t, out):
    out[:] = theta[:]


def prop_x(reaction, X, out):
    if reaction == 0:
        out[:] = X[:, 0]
        return None
    if reaction == 1:
        out[:] = X[:, 1]
        return None
    if reaction == 2:
        out[:] = X[:, 1]
        return None
    if reaction == 3:
        out[:] = X[:, 2]
        return None


init_bounds = np.array([1, 1, 20])


comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()
#
# solver = SensFspSolverMultiSinks(comm)
# solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
# solver.SetVerbosity(2)
# solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
# solver.SetInitialDist(np.array(X0), np.array(P0), [np.array(S0)] * 4)
# solutions = solver.SolveTspan(t_meas, 1.0e-8)
#
# rna_distributions = []
# rna_sensitivities = []
# for i in range(0, len(solutions)):
#     rna_distributions.append(solutions[i].Marginal(2))
#     sens_list = []
#     for iS in range(0, 4):
#         sens_list.append(solutions[i].SensMarginal(iS, 2))
#     rna_sensitivities.append(sens_list)
#
# if rank == 0:
#     np.savez('bursting_parameters.npz', kon=k_on, koff = k_off, kr= k_r, gamma=gamma)
#     np.savez('fsp_solutions.npz', rna_distributions=rna_distributions,
#          rna_sensitivities=rna_sensitivities, t_meas=t_meas, allow_pickle=True)
#%%
if rank == 0:
    with np.load('fsp_solutions.npz', allow_pickle=True) as f:
        rna_distributions=f['rna_distributions']
        rna_sensitivities=f['rna_sensitivities']
        t_meas = f['t_meas']
else:
    rna_distributions = None
    rna_sensitivities = None
    t_meas = None

rna_distributions = comm.bcast(rna_distributions)
rna_sensitivities = comm.bcast(rna_sensitivities)
t_meas = comm.bcast(t_meas)
#%%
if rank == 0:
# FIM for exact smFISH measurements
    fim_exact = np.zeros((len(t_meas), 4, 4))
    for itime in range(0, len(t_meas)):
        M = np.zeros((4, 4))
        for ip in range(0, 4):
            for jp in range(0, ip + 1):
                M[ip, jp] = np.sum(rna_sensitivities[itime][ip] * rna_sensitivities[itime][jp] /
                                   np.maximum(rna_distributions[itime], 1.0e-16))
        for ip in range(0, 4):
            for jp in range(ip + 1, 4):
                M[ip, jp] = M[jp, ip]
        fim_exact[itime, :, :] = M
    np.savez('fim_exact.npz', fim_exact=fim_exact)
#%%
# FIM for flow cytometry measurements
from chebpy import chebfun
kappa = 220
sigma_probe = 300
mu_bg = 100
sigma_bg = 200
#%%
if rank == 0:
    np.savez('flowcyt_pars.npz', kappa=kappa, sigma_probe=sigma_probe, mu_bg=mu_bg, sigma_bg=sigma_bg)
#%%
# This function computes the conditional probability density of the intensity given the number of mRNA molecules
def intensitycondprob(y, x):
    return np.exp(-((y - x*kappa - mu_bg)**2.0)/(2.0*(x*x*sigma_probe**2.0+sigma_bg**2.0)))/ \
    np.sqrt(2*np.pi*(x*x*sigma_probe**2.0+sigma_bg**2.0))

# Function to evaluate the pointwise probability density of the intensity
def intensitypointwise(y, p_x):
    xrange = np.arange(0, len(p_x))
    fval = np.zeros((len(y),))
    for i in range(0, len(y)):
        fval[i] = np.dot(intensitycondprob(y[i], xrange), p_x)
    return fval

# Accuracy of the quadrature
npoints_per_cpu = 10000
npoints = ncpus*npoints_per_cpu

fc_prob_evals = []
fc_sens_evals = []
for itime in range(0, len(t_meas)):
    if rank == 0:
        print(itime)
    xmax = len(rna_distributions[itime])-1

    yrange = [mu_bg - 3*sigma_bg, kappa*200 + 3*sigma_bg]
    yeval = np.linspace(yrange[0], yrange[1], npoints)

    yeval_loc = yeval[rank*npoints_per_cpu:(rank+1)*npoints_per_cpu]
    peval_loc = intensitypointwise(yeval_loc, rna_distributions[itime])

    if rank == 0:
        peval = np.zeros((len(yeval),), dtype=float)
        counts = npoints_per_cpu*np.ones((ncpus,), dtype=int)
        displs = np.zeros((ncpus,))
        displs[1:] = np.cumsum(counts[0:-1])
    else:
        peval = None
        counts = None
        displs = None

    comm.Gatherv(sendbuf=[peval_loc, npoints_per_cpu, mpi.DOUBLE], recvbuf=[peval, counts, displs, mpi.DOUBLE])

    if rank == 0:
        fc_prob_evals.append({'yeval': yeval, 'peval': peval})

    stmp = []
    for ip in range(0, 4):
        seval_loc = intensitypointwise(yeval_loc, rna_sensitivities[itime][ip])

        if rank == 0:
            seval = np.zeros((len(yeval),), dtype=float)
            counts = npoints_per_cpu * np.ones((ncpus,), dtype=int)
            displs = np.zeros((ncpus,))
            displs[1:] = np.cumsum(counts[0:-1])
        else:
            seval = None
            counts = None
            displs = None

        comm.Gatherv(sendbuf=[seval_loc, npoints_per_cpu, mpi.DOUBLE], recvbuf=[seval, counts, displs, mpi.DOUBLE])

        if rank == 0:
            stmp.append({'yeval': yeval, 'seval': seval})
    if rank == 0:
        fc_sens_evals.append(stmp)
#%%
if rank == 0:
    fim_flowcyt = np.zeros((len(t_meas), 4, 4))
    stepsize = (yeval[-1] - yeval[0])/len(yeval)
    for itime in range(0, len(t_meas)):
        M = np.zeros((4,4))
        print(itime)

        for ip in range(0, 4):
            for jp in range(0, ip + 1):
                M[ip, jp] = stepsize*np.sum(fc_sens_evals[itime][ip]['seval']*fc_sens_evals[itime][jp][
                    'seval']/fc_prob_evals[
                    itime]['peval'])
        for ip in range(0,4):
            for jp in range(ip+1,4):
                M[ip, jp] = M[jp, ip]
        fim_flowcyt[itime, :, :] = M
#%%
if rank == 0:
    np.savez('fim_flowcyt.npz', fim_flowcyt=fim_flowcyt, flowcyt_prob=fc_prob_evals, flowcyt_sens=fc_sens_evals)
#%%
