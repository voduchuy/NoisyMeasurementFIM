import pypacmensl
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
from mapk_model import YeastModel, _DEFAULT_PARAMETERS, fsp_constraints

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()

#%% Simulate the stationary solution
model = YeastModel(osmotic=False)
sfsp = SensFspSolverMultiSinks(comm)
sfsp.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=model.propensity_t,
    propensity_x=model.propensity_x,
    tv_reactions=[1],
    d_propensity_t=model.dpropt,
    d_propensity_t_sp=model.dpropt_sparsity,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity,
)
sfsp.SetVerbosity(2)
sfsp.SetFspShape(constr_fun=fsp_constraints, constr_bound=model.init_bounds)
sfsp.SetInitialDist(
    np.array(model.X0), np.array(model.P0), [np.array(model.S0)] * model.NUM_PARAMETERS
)
solution0 = sfsp.Solve(24 * 3600, 1.0e-2)
#%% Simulate the 1 hour with osmotic shock where finest smapling time is 1 minute
model = YeastModel(osmotic=True)
t_meas = np.linspace(0, 1 * 3600, 1 * 3600 // 60 + 1)
sfsp = SensFspSolverMultiSinks(comm)
sfsp.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=model.propensity_t,
    propensity_x=model.propensity_x,
    tv_reactions=[1],
    d_propensity_t=model.dpropt,
    d_propensity_t_sp=model.dpropt_sparsity,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity,
)
sfsp.SetVerbosity(2)
sfsp.SetFspShape(constr_fun=fsp_constraints, constr_bound=model.init_bounds)
sfsp.SetInitialDist1(solution0)
solutions = sfsp.SolveTspan(t_meas, 1.0e-6)
#%% Compute the FIM for full joint measurements that include gene state
fim = np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
for i, v in enumerate(solutions):
    fim[i, ...] = v.ComputeFIM()

if rank == 0:
    np.savez("results/fim_exact.npz", fim=fim)
#%% FIMs for marginal measurements
fim_marginals = [
    np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS)),
    np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS)),
]
for ispecies, species in enumerate([4, 5]):
    for itime in range(len(solutions)):
        marginal_sens = [
            solutions[itime].SensMarginal(i, species)
            for i in range(model.NUM_PARAMETERS)
        ]
        marginal_dist = solutions[itime].Marginal(species)
        for ipar in range(model.NUM_PARAMETERS):
            for jpar in range(ipar + 1):
                fij = np.sum(
                    marginal_sens[ipar]
                    * marginal_sens[jpar]
                    / (marginal_dist + 1.0e-16)
                )
                fim_marginals[ispecies][itime, ipar, jpar] = fij

        for ipar in range(model.NUM_PARAMETERS):
            fim_marginals[ispecies][itime, ipar, ipar + 1 :] = fim_marginals[ispecies][
                itime, ipar + 1 :, ipar
            ]
if rank == 0:
    np.savez(
        "results/fim_marginals.npz",
        nuc_only=fim_marginals[0],
        cyt_only=fim_marginals[1],
    )
#%% FIMs for the total mRNA measurements
nrna_max = 200

def total_rna_pdo(x, py):
    py[:] = 0.0
    idx = min(x[4] + x[5], nrna_max)
    py[idx] = 1.0

fim_total_rna = np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
for itime in range(len(solutions)):
    py_sens = [
        solutions[itime].WeightedAverage(i, nrna_max + 1, total_rna_pdo)
        for i in range(model.NUM_PARAMETERS)
    ]
    py = solutions[itime].WeightedAverage(-1, nrna_max + 1, total_rna_pdo)
    for ipar in range(model.NUM_PARAMETERS):
        for jpar in range(ipar + 1):
            fij = np.sum(py_sens[ipar] * py_sens[jpar] / (py + 1.0e-8))
            fim_total_rna[itime, ipar, jpar] = fij

    for ipar in range(model.NUM_PARAMETERS):
        fim_total_rna[itime, ipar, ipar + 1 :] = fim_total_rna[itime, ipar + 1 :, ipar]

if rank == 0:
    np.savez("results/fim_total_rna.npz", fim=fim_total_rna)
#%% FIMs for joint measurements without gene states
nuc_max = 20
cyt_max = 100
nstates2d = (nuc_max+1)*(cyt_max+1)

def joint_rna_pdo(x, py):
    py[:] = 0.0
    idx = min([x[5], cyt_max]) + min([x[4], nuc_max])*(cyt_max+1)
    py[idx] = 1.0

fim_joint_rna = np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
for itime in range(len(solutions)):
    py_sens = [
        solutions[itime].WeightedAverage(i, nstates2d + 1, joint_rna_pdo)
        for i in range(model.NUM_PARAMETERS)
    ]
    py = solutions[itime].WeightedAverage(-1, nstates2d + 1, joint_rna_pdo)
    for ipar in range(model.NUM_PARAMETERS):
        for jpar in range(ipar + 1):
            fij = np.sum(py_sens[ipar] * py_sens[jpar] / (py + 1.0e-8))
            fim_joint_rna[itime, ipar, jpar] = fij

    for ipar in range(model.NUM_PARAMETERS):
        fim_joint_rna[itime, ipar, ipar + 1 :] = fim_joint_rna[itime, ipar + 1 :, ipar]

if rank == 0:
    np.savez("results/fim_joint_rna.npz", fim=fim_joint_rna)
