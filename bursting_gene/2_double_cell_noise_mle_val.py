import sys
from typing import List
from bursting_gene_model import BurstingGeneModel
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
import mpi4py.MPI as mpi
import numpy as np
import pygmo
from typing import Union
from distortion_models import DistortionModel, ZeroDistortion, DoubleCell


def simulateExactData(
    theta,
    t_meas: np.ndarray,
    ncells: int = 1000,
):
    """Simulate exact SmFish observations
    Arguments
    ---------
    theta: 1-D array
        Array of parameters for bursting gene model, ordered as (k01, k10, transcription_rate, degradation_rate).

    t_meas: 1-D array
        Array of measurement times.

    ncells: int
        Number of cells per measurment time.

    Returns
    -------
    data: List of 1-D arrays
        List of measured mRNA copy numbers at the input measurement times.
    """
    model = BurstingGeneModel(theta)
    ssa = SSASolver(mpi.COMM_SELF)
    ssa.SetModel(model.stoichMatrix, None, model.propensity_x)
    data = []
    for t in t_meas:
        X = ssa.Solve(t, model.X0, ncells, send_to_root=True)
        y = X[:, 2]
        data.append(y)
    return data


def simulateDoubledData(
    theta,
    t_meas: np.ndarray,
    ncells: int = 1000,
):
    """Simulate distorted SmFish observations where each observation consists of the sum of two independent cells.

    Arguments
    ---------
    theta: 1-D array
        Array of parameters for bursting gene model, ordered as (k01, k10, transcription_rate, degradation_rate).

    t_meas: 1-D array
        Array of measurement times.

    ncells: int
        Number of cells per measurment time.

    Returns
    -------
    data: List of 1-D arrays
        List of measured mRNA copy numbers at the input measurement times.
    """
    model = BurstingGeneModel(theta)
    ssa = SSASolver(mpi.COMM_SELF)
    ssa.SetModel(model.stoichMatrix, None, model.propensity_x)
    data = []
    for t in t_meas:
        X1 = ssa.Solve(t, model.X0, ncells, send_to_root=True)
        X2 = ssa.Solve(t, model.X0, ncells, send_to_root=True)
        y = X1[:, 2] + X2[:, 2]
        data.append(y)
    return data


def solveCme(log10_theta, t_meas):
    theta = np.power(10.0, log10_theta)
    model = BurstingGeneModel(theta)
    cme_solver = FspSolverMultiSinks(mpi.COMM_SELF)
    cme_solver.SetVerbosity(0)
    cme_solver.SetModel(model.stoichMatrix, None, model.propensity_x)
    cme_solver.SetFspShape(constr_fun=None, constr_bound=np.array([1, 1, 20]))
    cme_solver.SetInitialDist(model.X0, model.P0)
    cme_solver.SetUp()
    solutions = cme_solver.SolveTspan(t_meas, 1.0e-6)
    cme_solver.ClearState()
    return solutions


def solveSensCme(log10_theta, t_meas):
    theta = np.power(10.0, log10_theta)
    model = BurstingGeneModel(theta)
    cme_solver = SensFspSolverMultiSinks(mpi.COMM_SELF)
    cme_solver.SetVerbosity(0)
    cme_solver.SetModel(
        num_parameters=4,
        stoich_matrix=np.array(model.stoichMatrix),
        propensity_t=None,
        propensity_x=model.propensity_x,
        tv_reactions=[],
        d_propensity_t=None,
        d_propensity_t_sp=None,
        d_propensity_x=model.dpropensity_x,
        d_propensity_x_sp=model.dpropx_sparsity,
    )
    cme_solver.SetFspShape(constr_fun=None, constr_bound=np.array([1, 1, 20]))
    cme_solver.SetInitialDist(model.X0, model.P0)
    cme_solver.SetUp()
    solutions = cme_solver.SolveTspan(t_meas, 1.0e-8)
    cme_solver.ClearState()
    return solutions


def negLoglike(
    log10_theta: np.ndarray,
    t_meas: np.ndarray,
    smfishdata,
    distortion: Union[DistortionModel, DoubleCell],
):
    try:
        solutions = solveCme(log10_theta, t_meas)
    except:
        return 1.0e8

    ll = 0.0
    for i in range(0, len(t_meas)):
        observations = smfishdata[i]

        xmax = max(observations)

        pvec_rna = solutions[i].Marginal(2)
        pvec_distorted = distortion.transformDistribution(pvec_rna)
        pvec_distorted = np.pad(
            pvec_distorted,
            (0, max([xmax + 1, len(pvec_distorted)]) - len(pvec_distorted)),
            constant_values=0.0,
        )
        ll += np.sum(np.log(pvec_distorted[observations] + 1.0e-16))
    return -1.0 * ll


def negLoglikeGradient(
    log10_theta: np.ndarray,
    t_meas: np.ndarray,
    smfishdata,
    distortion: Union[DistortionModel, DoubleCell],
):
    try:
        solutions = solveSensCme(log10_theta, t_meas)
    except:
        return np.random.uniform(size=4)

    ll = 0.0
    ll_grad = np.zeros((4,))
    for i in range(0, len(t_meas)):
        observations = smfishdata[i]

        xmax = max(observations)

        pvec_rna = solutions[i].Marginal(2)
        svecs_rna = [solutions[i].SensMarginal(ip, 2) for ip in range(4)]

        pvec_distorted = distortion.transformDistribution(pvec_rna)
        pvec_distorted = np.pad(
            pvec_distorted,
            (0, max([xmax + 1, len(pvec_distorted)]) - len(pvec_distorted)),
            constant_values=0.0,
        )
        svecs_distorted = [distortion.transformDistribution(svec) for svec in svecs_rna]
        svecs_distorted = [
            np.pad(
                svec, (0, max([xmax + 1, len(svec)]) - len(svec)), constant_values=0.0
            )
            for svec in svecs_distorted
        ]

        grad_here = [
            np.sum(svec[observations] / (pvec_distorted[observations] + 1.0e-16)) for svec in svecs_distorted
        ] / (10.0**log10_theta)
        ll_grad += np.array(grad_here)
    return -1.0 * ll_grad


class PyGmoOptProblem:
    def __init__(
        self,
        data: List[np.ndarray],
        t_meas: np.ndarray,
        distortion: DistortionModel,
        par_lb: np.ndarray,
        par_ub: np.ndarray,
    ):
        self.dim = 4
        self.data = data
        self.t_meas = t_meas
        self.distortion = distortion
        self.lb = par_lb
        self.ub = par_ub
        self.rank = mpi.COMM_WORLD.Get_rank()

    def get_bounds(self):
        return (self.lb, self.ub)

    def fitness(self, dv):
        ll = negLoglike(dv, self.t_meas, self.data, self.distortion)
        return [ll]

    def gradient(self, x):
        grad = negLoglikeGradient(x, self.t_meas, self.data, self.distortion)
        print(grad)
        return grad




def mleFit(datasets, distortion_model):
    num_datasets_local = len(datasets)
    fits_local = np.zeros((num_datasets_local, 4))
    for itrial in range(0, num_datasets_local):
        print(f"FITTING DATASET {itrial}...")
        data = datasets[itrial]
        prob = pygmo.problem(
            PyGmoOptProblem(
                data=data,
                t_meas=T_MEAS,
                distortion=distortion_model,
                par_lb=log10theta_lb,
                par_ub=log10theta_ub,
            )
        )
        pop = pygmo.population(prob)
        pop.push_back(np.log10(theta_true))
        start_range0 = 0.01
        my_algo = pygmo.compass_search(
            max_fevals=20000, start_range=start_range0, stop_range=1.0e-5
        )
        # my_algo = pygmo.scipy_optimize(method="L-BFGS-B", tol=1.0E-7)
        algo = pygmo.algorithm(my_algo)
        algo.set_verbosity(1)
        pop = algo.evolve(pop)
        fits_local[itrial, :] = pop.champion_x

    if RANK == 0:
        fits_all = np.zeros((dataset_count, NUM_PARAMETERS))

        buffersizes = (
                NUM_PARAMETERS * (dataset_count // NPROCS) * np.ones((NPROCS,), dtype=int)
        )
        buffersizes[0 : (dataset_count % NPROCS)] += NUM_PARAMETERS

        displacements = np.zeros((NPROCS,), dtype=int)
        displacements[1:] = np.cumsum(buffersizes[0:-1])
    else:
        fits_all = None
        buffersizes = None
        displacements = None

    mpi.COMM_WORLD.Gatherv(
        sendbuf=fits_local,
        recvbuf=[fits_all, buffersizes, displacements, mpi.DOUBLE],
        root=0,
    )
    return fits_all


if __name__ == "__main__":
    RANK = mpi.COMM_WORLD.Get_rank()
    NPROCS = mpi.COMM_WORLD.Get_size()
    ARGV = sys.argv
    # %% Options for the optimization run
    options = {
        "cell_count": 100,
        "dataset_count": 100
    }
    # %% Parse command line arguments
    for i in range(1, len(ARGV)):
        key, value = ARGV[i].split("=")
        if key in options:
            options[key] = int(value)
        else:
            print(f"WARNING: Unknown option {key} \n")

    dataset_count = options["dataset_count"]
    cell_count = options["cell_count"]
#%%
    rng = np.random.default_rng(RANK)
    T_MEAS = 30.0 * np.arange(1, 6)
    NUM_PARAMETERS = 4

    true_model = BurstingGeneModel()
    theta_true = np.array(
        [true_model.k01, true_model.k10, true_model.alpha, true_model.gamma]
    )
    theta_lb = 0.001 * theta_true
    theta_ub = 1000 * theta_true
    log10theta_lb = np.log10(theta_lb)
    log10theta_ub = np.log10(theta_ub)

    #%%
    # Simulate distorted_datasets (with distorted measurements) and perform fits to the distorted data using the corrected likelihood function
    distortion_model = DoubleCell(rho=1.0)
    local_dataset_count = dataset_count // NPROCS + (RANK < dataset_count % NPROCS)
    distorted_datasets = []
    for itrial in range(0, local_dataset_count):
        distorted_datasets.append(
            simulateDoubledData(
                theta_true,
                t_meas=T_MEAS,
                ncells=cell_count,
            )
        )
    distorted_data_fits = mleFit(distorted_datasets, distortion_model)
    #%%
    # Simulate distorted_datasets (with distorted measurements) and perform fits to the distorted data using the corrected likelihood function
    exact_datasets = []
    for itrial in range(0, local_dataset_count):
        exact_datasets.append(
            simulateExactData(
                theta_true,
                t_meas=T_MEAS,
                ncells=cell_count,
            )
        )
    exact_data_fits = mleFit(exact_datasets, ZeroDistortion())
    if RANK == 0:
        np.savez(
            "results/double_cell_mle_fits.npz",
            distorted_data_fits=distorted_data_fits,
            exact_data_fits=exact_data_fits,
        )
