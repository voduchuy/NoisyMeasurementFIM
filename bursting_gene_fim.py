from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np

k_off = 0.015
k_on = 0.05
k_r = 5
gamma = 0.05

t_meas = np.linspace(0, 400, 401)

theta = np.array([k_on, k_off, k_r, gamma])

SM = [[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]]
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


comm = mpi.COMM_SELF
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(np.array(SM), prop_t, prop_x, dprop_t_list, [prop_x] * 4, dprop_sparsity)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(np.array(X0), np.array(P0), [np.array(S0)] * 4)
solutions = solver.SolveTspan(t_meas, 1.0e-8)

rna_distributions = []
rna_sensitivities = []
for i in range(0, len(solutions)):
    rna_distributions.append(solutions[i].Marginal(2))
    sens_list = []
    for iS in range(0, 4):
        sens_list.append(solutions[i].SensMarginal(iS, 2))
    rna_sensitivities.append(sens_list)
np.savez('bursting_parameters.npz', kon=k_on, koff = k_off, kr= k_r, gamma=gamma)
np.savez('fsp_solutions.npz', rna_distributions=rna_distributions,
         rna_sensitivities=rna_sensitivities, t_meas=t_meas, allow_pickle=True)
#%%
with np.load('fsp_solutions.npz', allow_pickle=True) as f:
    rna_distributions=f['rna_distributions']
    rna_sensitivities=f['rna_sensitivities']
    t_meas = f['t_meas']
#%%
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

flowcyt_intensity_prob = []
flowcyt_intensity_sens = []
fc_prob_evals = []
for itime in range(0, len(t_meas)):
    xmax = len(rna_distributions[itime])-1
    yrange = [mu_bg - 3*sigma_bg, kappa*xmax + mu_bg + 3*sigma_bg]
    flowcyt_intensity_prob.append(chebfun(lambda y: intensitypointwise(y, rna_distributions[itime]), yrange))
    stmp = []
    for ip in range(0, 4):
        stmp.append(chebfun(lambda y: intensitypointwise(y, rna_sensitivities[itime][ip]), yrange))
    flowcyt_intensity_sens.append(stmp)

    yeval = np.linspace(yrange[0], yrange[1], 200)
    peval = flowcyt_intensity_prob[itime](yeval)
    fc_prob_evals.append({'yeval': yeval, 'peval': peval})

fim_flowcyt = np.zeros((len(t_meas), 4, 4))
for itime in range(0, len(t_meas)):
    M = np.zeros((4,4))
    print(itime)
    # g = []
    # for ip in range(0, 4):
    #     g.append(flowcyt_intensity_sens[itime][ip]/
    #                            flowcyt_intensity_prob[itime].maximum(1.0e-18))
    # for ip in range(0, 4):
    #     for jp in range(0, ip+1):
    #         M[ip, jp] = (g[ip]).sum()

    for ip in range(0, 4):
        for jp in range(0, ip + 1):
            g = (flowcyt_intensity_sens[itime][ip] * flowcyt_intensity_sens[itime][jp] /
                     flowcyt_intensity_prob[itime].maximum(1.0e-10))
            M[ip, jp] = g.sum()
    for ip in range(0,4):
        for jp in range(ip+1,4):
            M[ip, jp] = M[jp, ip]
    fim_flowcyt[itime, :, :] = M
#%%
np.savez('fim_flowcyt.npz', fim_flowcyt=fim_flowcyt, flowcyt_prob=fc_prob_evals)
#%%
