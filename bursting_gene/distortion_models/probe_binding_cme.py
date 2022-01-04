from .base import DistortionModel
import numpy as np
from pypacmensl.fsp_solver.multi_sinks import FspSolverMultiSinks
import mpi4py.MPI as mpi

_PROBE_BINDING = 1.0E-1
_PROBE_UNBINDING = 1.0E-2
_CLUMPING_RATE = 1.0E-1
_CLUMP_DISSOLUTION = 5.0E-2
_PROBE_CONCENTRATION = 1.0E0

class ProbeBindingCme(object):
    stoich_matrix = np.array([[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]])

    def __init__(
        self,
        num_rna_init: int = 0,
        probe_binding: float = 1.0,
        probe_unbinding: float = 1.0,
        clumping_rate: float = 1.0,
        clump_dissolution: float = 1.0,
        probe_concentration: float = 1.0,
    ):
        self.n0 = num_rna_init
        self.probe_binding = probe_binding
        self.probe_unbinding = probe_unbinding
        self.clumping_rate = clumping_rate
        self.clump_dissolution = clump_dissolution
        self.probe_concentration = probe_concentration

        self.x0 = np.array([self.n0, 0, 0], dtype=np.intc)

        def propensity_x(reaction, x, out):
            if reaction == 0:
                out[:] = self.probe_binding * self.probe_concentration * x[:, 0]
            elif reaction == 1:
                out[:] = self.probe_unbinding * x[:, 1]
            elif reaction == 2:
                out[:] = self.clumping_rate * self.probe_concentration
            elif reaction == 3:
                out[:] = self.clump_dissolution * x[:, 2]

        self.propensity_t = None
        self.propensity_x = propensity_x


#%%
class SmFishDistortion(DistortionModel):
    def __init__(
        self,
        probe_binding: float = _PROBE_BINDING,
        probe_unbinding: float = _PROBE_UNBINDING,
        clumping_rate: float = _CLUMPING_RATE,
        clump_dissolution: float = _CLUMP_DISSOLUTION,
        probe_concentration: float = _PROBE_CONCENTRATION,
        comm: mpi.Comm = mpi.COMM_WORLD
    ):
        super().__init__()
        self.probe_binding = probe_binding
        self.probe_unbinding = probe_unbinding
        self.clumping_rate = clumping_rate
        self.clump_dissolution = clump_dissolution
        self.probe_concentration = probe_concentration
        self.comm = comm

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        fsp = FspSolverMultiSinks(mpi.COMM_SELF)
        cme = ProbeBindingCme(
            num_rna_init=x,
            probe_binding=self.probe_binding,
            probe_unbinding=self.probe_unbinding,
            clumping_rate=self.clumping_rate,
            clump_dissolution=self.clump_dissolution,
            probe_concentration=self.probe_concentration,
        )
        fsp.SetModel(stoich_matrix=cme.stoich_matrix,
                     propensity_x=cme.propensity_x,
                     propensity_t=None,
                     tv_reactions=[])
        print(f"Generating conditional probability vector for x={x}")
        fsp.SetInitialDist(cme.x0.reshape((1, 3)), np.array([1.0]))
        fsp.SetFspShape(None, np.array([x+1, 1, 1]))
        fsp.SetVerbosity(0)
        fsp.SetOdeSolver("PETSC")
        p = fsp.Solve(3.0E2, 1.0E-6)

        def wf(x, out):
            out[:] = 0.0
            i = x[1] + x[2]
            out[i] = 1.0

        pcond = p.WeightedAverage(len(y), wf)
        return pcond

    def getDenseMatrix(self, xrange: np.ndarray, yrange: np.ndarray) -> np.ndarray:
        nx = len(xrange)
        ny = len(yrange)
        M = np.zeros((ny, nx))

        for j, x in enumerate(xrange):
            M[:, j] = self.getConditionalProbabilities(x, yrange)
        return M


