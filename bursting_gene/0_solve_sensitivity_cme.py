from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
from bursting_gene_model import BurstingGeneModel

model = BurstingGeneModel()
t_meas = np.linspace(0, 400, 401)

solver = SensFspSolverMultiSinks(mpi.COMM_SELF)
solver.SetModel(
    num_parameters=4,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_x=model.propensity_x,
    d_propensity_x=model.dpropensity_x,
    d_propensity_x_sp=model.dpropx_sparsity,
)
solver.SetVerbosity(1)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist(np.array(model.X0), np.array(model.P0), [np.array(model.S0)] * 4)
solutions = solver.SolveTspan(t_meas, 1.0e-8)

rna_distributions = []
rna_sensitivities = []
for i in range(0, len(solutions)):
    rna_distributions.append(solutions[i].Marginal(2))
    sens_list = []
    for iS in range(0, 4):
        sens_list.append(solutions[i].SensMarginal(iS, 2))
    rna_sensitivities.append(sens_list)

np.savez(
    "results/bursting_parameters.npz",
    kon=model.k01,
    koff=model.k10,
    alpha=model.alpha,
    gamma=model.gamma,
)
np.savez(
    "results/fsp_solutions.npz",
    rna_distributions=rna_distributions,
    rna_sensitivities=rna_sensitivities,
    t_meas=t_meas,
)
