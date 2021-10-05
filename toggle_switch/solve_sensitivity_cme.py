from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
from toggle_model import ToggleSwitchModel

model = ToggleSwitchModel()

t_meas = np.linspace(0, 8*3600, 8*3600//60 + 1)

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()

solver = SensFspSolverMultiSinks(comm)
solver.SetModel(
    np.array(model.stoichMatrix),
    model.propensity_t,
    model.propensity_x,
    model.dprop_t_list,
    model.dprop_x_list,
    model.dprop_sparsity,
)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist(np.array(model.X0), np.array(model.P0), [np.array(model.S0)] * 10)
solutions = solver.SolveTspan(t_meas, 1.0e-8)

#%% FIM for full measurements of two species
fim = np.zeros((len(t_meas), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
for i, v in enumerate(solutions):
    fim[i, ...] = v.ComputeFIM()

if rank == 0:
    np.savez(
        "results/fim_exact.npz",
        fim=fim
    )
#%% FIM for marginal measurements
fim_marginals = [
    np.zeros((len(t_meas), model.NUM_PARAMETERS, model.NUM_PARAMETERS)),
    np.zeros((len(t_meas), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
]
for species in [0,1]:
    for itime in range(len(t_meas)):
        marginal_sens = [solutions[itime].SensMarginal(i, species) for i in range(model.NUM_PARAMETERS)]
        marginal_dist = solutions[itime].Marginal(species)
        for ipar in range(model.NUM_PARAMETERS):
            for jpar in range(ipar+1):
                fim_marginals[species][itime, ipar, jpar] = np.sum(marginal_sens[ipar]*marginal_sens[jpar]/(marginal_dist + 1.0E-16))
        for ipar in range(model.NUM_PARAMETERS):
            fim_marginals[species][itime, ipar, ipar+1:] = fim_marginals[species][itime, ipar+1:, ipar]
if rank == 0:
    np.savez(
        "results/fim_marginals.npz",
        fim0 = fim_marginals[0],
        fim1 = fim_marginals[1]
    )








