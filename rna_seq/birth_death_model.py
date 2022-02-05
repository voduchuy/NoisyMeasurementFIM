import numpy as np
import numpy.typing as npt
from scipy.stats import poisson
from pypacmensl.sensitivity.multi_sinks import SensFspSolverMultiSinks
import mpi4py.MPI as mpi


def poissonStationary(
    birth_rate: float, death_rate: float, n: int = 1000
) -> [np.ndarray, np.ndarray, np.ndarray]:
    def inverse_factorial(n: int):
        return np.prod(1.0 / np.arange(1, n + 1))

    krange = np.arange(0, n)
    prate = birth_rate / death_rate

    pvec = poisson.pmf(k=krange, mu=prate)
    svec_birth = pvec * ((1.0 / birth_rate) * krange - 1.0 / death_rate)
    svec_death = pvec * (
        birth_rate * death_rate ** (-2.0) - krange * (1.0 / death_rate)
    )

    return pvec, svec_birth, svec_death


class BirthDeath(object):
    def __init__(self, birth: float = 1.0, death: float = 1.0):
        self.birth = birth
        self.death = death
        self.stoich_matrix = np.array([[1], [-1]])

        t_halflife = 1800 # We will let birth rate be reduced by half in 30 minutes
        ln2 = np.log(2)

        def propensity_tv(t: float, out: np.ndarray):
            out[0] = np.exp(-ln2 * t/t_halflife)

        def propensity(reaction: int, x: np.ndarray, out: np.ndarray):
            if reaction == 0:
                out[:] = self.birth
            else:
                out[:] = self.death * x[:, 0]
            return None

        def dpropensity(par: int, reaction: int, x: np.ndarray, out: np.ndarray):
            if par == 0:
                if reaction == 0:
                    out[:] = 1.0
            elif par == 1:
                if reaction == 1:
                    out[:] = x[:, 0]
            return None

        self.propensity_tv = propensity_tv
        self.dpropensity_tv = None
        self.dpropensity_t_sp = None
        self.propensity = propensity
        self.dpropensity = dpropensity
        self.dpropensity_sp = [[0], [1]]

        n = 100
        while True:
            p0, s0_b, s0_d = poissonStationary(self.birth, self.death, n=n)
            if 1.0 - np.sum(p0) > 1.0e-10:
                n += 50
            else:
                break
        self.x0 = np.arange(0, n).reshape((n, 1))
        self.p0 = p0
        self.s0 = [s0_b, s0_d]
        self.init_fsp = np.array([[n]])

    def generateSFspSolver(self):
        sfsp = SensFspSolverMultiSinks(mpi.COMM_SELF)
        sfsp.SetFspShape(None, self.init_fsp)
        sfsp.SetModel(
            2,
            stoich_matrix=self.stoich_matrix,
            propensity_t=self.propensity_tv,
            propensity_x=self.propensity,
            tv_reactions=[0],
            d_propensity_t=self.dpropensity_tv,
            d_propensity_t_sp=self.dpropensity_t_sp,
            d_propensity_x=self.dpropensity,
            d_propensity_x_sp=self.dpropensity_sp,
        )
        sfsp.SetVerbosity(0)
        sfsp.SetInitialDist(self.x0, self.p0, self.s0)
        return sfsp


def sensfspBirthDeathStationary(
    birth_rate: float, death_rate: float, t: float = 1e4
) -> [np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.array([[0]])
    stoich_matrix = np.array([[1], [-1]])
    init_bound = np.array([100])

    def propensity(reaction: int, x: np.ndarray, out: np.ndarray):
        if reaction == 0:
            out[:] = birth_rate
        else:
            out[:] = death_rate * x[:, 0]
        return None

    def dpropx(par: int, reaction: int, x: np.ndarray, out: np.ndarray):
        if par == 0:
            if reaction == 0:
                out[:] = 1.0
        elif par == 1:
            if reaction == 1:
                out[:] = x[:, 0]

    sfsp = SensFspSolverMultiSinks(mpi.COMM_SELF)
    sfsp.SetFspShape(None, init_bound)
    sfsp.SetModel(
        2,
        stoich_matrix=stoich_matrix,
        propensity_t=None,
        propensity_x=propensity,
        tv_reactions=[],
        d_propensity_t=None,
        d_propensity_t_sp=None,
        d_propensity_x=dpropx,
        d_propensity_x_sp=[[0], [1]],
    )
    sfsp.SetVerbosity(2)
    sfsp.SetInitialDist(x0, np.array([1.0]), [np.array([0.0]), np.array([0.0])])
    solution = sfsp.Solve(t, 1.0e-6)

    return (
        solution.Marginal(0),
        solution.SensMarginal(0, 0),
        solution.SensMarginal(1, 0),
    )


if __name__ == "__main__":
    from numpy.linalg import norm

    birth = 1000.0
    death = 10.0

    p, sb, sd = sensfspBirthDeathStationary(birth, death)
    pexact, sbexact, sdexact = poissonStationary(birth, death, n=len(p))

    print(norm(p - pexact, 1))
    print(norm(sb - sbexact, 1))
    print(norm(sd - sdexact, 1))
