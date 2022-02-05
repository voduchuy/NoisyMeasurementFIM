import sys

sys.path.append("..")
import numpy as np
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
from bursting_gene.distortion_models import BinomialDistortionModel
from bursting_gene.common_settings import computeSingleObservationFim
from birth_death_model import BirthDeath
import mpi4py.MPI as mpi
import pathlib as pl


def computePointwiseFim(birth: float, death: float, tau: float):
    model = BirthDeath(birth=birth, death=death)
    distortion = BinomialDistortionModel(detectionRate=0.35)
    sfsp = model.generateSFspSolver()
    solutions = sfsp.SolveTspan(np.array([0.0, tau, 2.0 * tau]), 1.0e-6)

    pvecs = [sol.Marginal(0) for sol in solutions]
    svecs = [(sol.SensMarginal(0, 0), sol.SensMarginal(1, 0)) for sol in solutions]
    xmax = max([len(v) for v in pvecs])
    C = distortion.getDenseMatrix(np.arange(xmax), np.arange(xmax))
    return computeSingleObservationFim(pvecs, svecs, C)


if __name__ == "__main__":
    # MPI communicator info
    comm: mpi.Comm = mpi.COMM_WORLD
    cpuid = comm.Get_rank()
    ncpus = comm.Get_size()

    # Choice for the experiment design variable tau: gap between two subsequent single-cell measurements
    tau_choices = [300, 3600, 7200]

    # Generate the global grid of birth and death rates
    nodes_per_axis = 5
    birth_nodes = np.logspace(-4.0, 0.0, nodes_per_axis)
    death_nodes = np.logspace(-4.0, -1.0, nodes_per_axis)
    bb, dd = np.meshgrid(birth_nodes, death_nodes, indexing="ij")

    # Compute the work distribution among CPUs
    nn_loc_layout = (nodes_per_axis // ncpus) * np.ones((ncpus,), dtype=int) + (
        np.arange(ncpus) < (nodes_per_axis % ncpus)
    )
    brange_loc_layout = np.zeros((ncpus,), dtype=int)
    brange_loc_layout[1:] = np.cumsum(nn_loc_layout[:-1])

    # Compute the local subset of fim matrices
    nn_loc = nn_loc_layout[cpuid]
    brange_loc = brange_loc_layout[cpuid]

    fims_loc = np.zeros(
        (nn_loc, nodes_per_axis, len(tau_choices), 3, 2, 2), dtype=np.double
    )
    for i in range(nn_loc):
        for j in range(nodes_per_axis):
            for it, tau in enumerate(tau_choices):
                print(i + brange_loc, j, it)
                fims_loc[i, j, it, :, :, :] = computePointwiseFim(
                    bb[i + brange_loc, j], dd[i + brange_loc, j], tau
                )

    # Gather back to root CPU
    sc = np.prod(fims_loc.shape[1:])
    if cpuid == 0:
        fims = np.zeros((nodes_per_axis, nodes_per_axis, len(tau_choices), 3, 2, 2))
        recv_buf = (fims, sc * nn_loc_layout, sc * brange_loc_layout, mpi.DOUBLE)
    else:
        recv_buf = None
    send_buf = (fims_loc, sc * nn_loc, mpi.DOUBLE)
    comm.Gatherv(send_buf, recv_buf)

    if cpuid == 0:
        if not pl.Path("./results").exists():
            pl.Path("./results").mkdir()

        np.savez(
            "results/fims.npz",
            bnodes=birth_nodes,
            dnodes=death_nodes,
            fims=fims,
            taus=tau_choices,
        )
