import pypacmensl
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
from three_state_model import ThreeStateModel, _DEFAULT_PARAMETERS, fsp_constraints

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()

#%% Simulate the stationary solution
parameters = _DEFAULT_PARAMETERS.copy()
parameters["b12"] = 0.0
model = ThreeStateModel(parameters)
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=model.propensity_t,
    propensity_x=model.propensity_x,
    tv_reactions=[2],
    d_propensity_t=model.dpropt,
    d_propensity_t_sp=model.dpropt_sparsity,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity
)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist(np.array(model.X0), np.array(model.P0), [np.array(model.S0)] * model.NUM_PARAMETERS)
solution0 = solver.Solve(24*3600, 1.0E-6)

