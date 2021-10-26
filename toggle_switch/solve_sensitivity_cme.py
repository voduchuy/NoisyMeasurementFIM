import pypacmensl
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
from toggle_model import ToggleSwitchModel

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()

#%% Simulate the stationary solution
model = ToggleSwitchModel()
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=None,
    propensity_x=model.propensity_x,
    tv_reactions=[],
    d_propensity_t=None,
    d_propensity_t_sp=None,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity
)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist(np.array(model.X0), np.array(model.P0), [np.array(model.S0)] * model.NUM_PARAMETERS)
solution0 = solver.Solve(24*3600, 1.0E-6)
## Simulate the next 2 hrs with pulsing
UV = 10
t_meas = np.linspace(0, 2*3600, 2*3600//60 + 1)
model = ToggleSwitchModel(UV=UV)
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=None,
    propensity_x=model.propensity_x,
    tv_reactions=[],
    d_propensity_t=None,
    d_propensity_t_sp=None,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity
)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist1(solution0)
solutions = solver.SolveTspan(t_meas, 1.0E-6)
## Simulate the remaining 6 hrs without pulsing
t_meas= np.linspace(0, 6*3600, 6*3600//60 + 1)
model = ToggleSwitchModel(UV=0.0)
solver = SensFspSolverMultiSinks(comm)
solver.SetModel(
    num_parameters=model.NUM_PARAMETERS,
    stoich_matrix=np.array(model.stoichMatrix),
    propensity_t=None,
    propensity_x=model.propensity_x,
    tv_reactions=[],
    d_propensity_t=None,
    d_propensity_t_sp=None,
    d_propensity_x=model.dpropx,
    d_propensity_x_sp=model.dpropx_sparsity
)
solver.SetVerbosity(2)
solver.SetFspShape(constr_fun=None, constr_bound=model.init_bounds)
solver.SetInitialDist1(solutions[-1])
solutions+=solver.SolveTspan(t_meas, 1.0E-6)
#%% FIM for full measurements of two species
fim = np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
for i, v in enumerate(solutions):
    fim[i, ...] = v.ComputeFIM()

if rank == 0:
    np.savez(
        "results/fim_exact.npz",
        fim=fim
    )
#%% FIM for marginal measurements
fim_marginals = [
    np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS)),
    np.zeros((len(solutions), model.NUM_PARAMETERS, model.NUM_PARAMETERS))
]
for species in [0,1]:
    for itime in range(len(solutions)):
        marginal_sens = [solutions[itime].SensMarginal(i, species) for i in range(model.NUM_PARAMETERS)]
        marginal_dist = solutions[itime].Marginal(species)
        for ipar in range(model.NUM_PARAMETERS):
            for jpar in range(ipar+1):
                fij = np.sum(marginal_sens[ipar] * marginal_sens[jpar] / (marginal_dist + 1.0E-16))
                fim_marginals[species][itime, ipar, jpar] = fij

        for ipar in range(model.NUM_PARAMETERS):
            fim_marginals[species][itime, ipar, ipar+1:] = fim_marginals[species][itime, ipar+1:, ipar]
if rank == 0:
    np.savez(
        "results/fim_marginals.npz",
        fim0 = fim_marginals[0],
        fim1 = fim_marginals[1]
    )

# %%
def compute_means(solution: pypacmensl.sensitivity.SensDiscreteDistribution)->np.ndarray:
    def fmean(x, out):
        out[0] = x[0]
        out[1] = x[1]
    output = np.zeros((22,))
    output[0:2] = solution.WeightedAverage(-1, 2, fmean)
    for i in range(10):
        output[2*(i+1):2*(i+1)+2] = solution.WeightedAverage(i, 2, fmean)
    return output

means = np.zeros((len(solutions), 22))
for i, solution in enumerate(solutions):
    means[i, :] = compute_means(solution)
# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 11*5), dpi=300, tight_layout=True)
axs = fig.subplots(11,1)
axs[0].plot(means[:,0], label="X")
axs[0].plot(means[:,1], label="Y")
axs[0].set_title("Mean")
for i in range(10):
    axs[i+1].plot(means[:, (i+1)*2])
    axs[i+1].plot(means[:, (i+1)*2 + 1])
    axs[i+1].set_title(f"Derivative of mean w.r.t. {i+1}th parameter")
fig.savefig("toggle_means.png", dpi=300, bbox_inches="tight")