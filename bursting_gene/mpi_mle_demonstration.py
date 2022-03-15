import sys
from typing import List
from bursting_gene_model import BurstingGeneModel
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
import mpi4py.MPI as mpi
import numpy as np
import pygmo
from distortion_models import DistortionModel, ZeroDistortion, BinomialVaryingDetectionRate


def simulateData(
    theta,
    t_meas: np.ndarray,
    ncells: int = 1000,
    distortion: DistortionModel = ZeroDistortion(),
    rng=np.random.default_rng(),
):
    """Simulate distorted SmFish observations
    Arguments
    ---------
    theta: 1-D array
        Array of parameters for bursting gene model, ordered as (k01, k10, transcription_rate, degradation_rate).

    t_meas: 1-D array
        Array of measurement times.

    ncells: int
        Number of cells per measurment time.

    distortion: DistortionModel
        Probabilistic Distortion Operator.

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
        y = distortion.distort(X[:, 2], rng=rng)
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


def negLoglike(
    log10_theta: np.ndarray, t_meas: np.ndarray, smfishdata, distortion: DistortionModel
):
    try:
        solutions = solveCme(log10_theta, t_meas)
    except:
        return 1.0e8

    ll = 0.0
    for i in range(0, len(t_meas)):
        observations = smfishdata[i]
        pvec_rna = solutions[i].Marginal(2)
        C = distortion.getDenseMatrix(
            xrange=np.arange(0, len(pvec_rna)), yrange=observations
        )
        pvec_observations = C @ pvec_rna
        ll += np.sum(np.log(pvec_observations + 1.0e-16))
    return -1.0 * ll


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
        return pygmo.estimate_gradient(
            lambda x: self.fitness(x), x
        )  # we here use the low precision gradient


def mleFit(datasets, distortion_model):
    num_datasets_local = len(datasets)
    for itrial in range(0, num_datasets_local):
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
    rng = np.random.default_rng(RANK)

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

    NUM_PARAMETERS = 4

    T_MEAS = 30.0*np.arange(1, 6)

    distortion_model = BinomialVaryingDetectionRate()
    true_model = BurstingGeneModel()
    theta_true = np.array(
        [true_model.k01, true_model.k10, true_model.alpha, true_model.gamma]
    )
    theta_lb = 0.001 * theta_true
    theta_ub = 1000 * theta_true
    log10theta_lb = np.log10(theta_lb)
    log10theta_ub = np.log10(theta_ub)

    # Simulate distorted_datasets (with distorted measurements)
    local_dataset_count = dataset_count // NPROCS + (RANK < dataset_count % NPROCS)
    fits_local = np.zeros((local_dataset_count, 4))
    datasets = []
    for itrial in range(0, local_dataset_count):
        datasets.append(simulateData(theta_true, t_meas=T_MEAS, ncells=cell_count, distortion=distortion_model, rng=rng))

    # Perform fits using the corrected likelihood function
    fits_correct_likelihood = mleFit(datasets, distortion_model)

    # Perform fits using the incorrect likelihood function
    fits_incorrect_likelihood = mleFit(datasets, ZeroDistortion())

    if RANK == 0:
        np.savez(
            "results/ge_mle_fits.npz",
            fits_correct=fits_correct_likelihood,
            fits_incorrect=fits_incorrect_likelihood,
        )
