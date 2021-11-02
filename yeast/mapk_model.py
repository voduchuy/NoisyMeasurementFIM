import numpy as np
from typing import Union, Dict

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "r1": 9.01E-4,
    "r2": 3.05E-4,
    "k01": 3.89E-2,
    "k10": 7.62E-3,
    "k12": 3.93E-5,
    "b12": 9.6E-3,
    "k21": 8.37E-3,
    "alpha0": 1.09E-4,
    "alpha1": 1.64E-5,
    "alpha2": 9.99e-01,
    "ktrans": 1.0,
    "delta": 5.67E-5
}

def fsp_constraints(X, out):
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 2]
    out[:, 3] = X[:, 3]
    out[:, 4] = X[:, 4]
    out[:, 5] = X[:, 3] + X[:, 4]

class ThreeStateModel:
    X0 = [[1, 0, 0, 0, 0]]
    P0 = [1.0]
    S0 = [0.0]
    NUM_PARAMETERS = 12

    init_bounds = np.array([1, 1, 1, 10, 10, 10])

    stoichMatrix = [
        [-1, 1, 0, 0, 0],
        [1, -1, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, -1, 1],
        [0, 0, 0, 0, -1]
    ]

    def __init__(self, parameters: Union[np.ndarray, dict] = _DEFAULT_PARAMETERS):
        if type(parameters) == dict:
            self.r1 = parameters["r1"]
            self.r2 = parameters["r2"]
            self.k01 = parameters["k01"]
            self.k10 = parameters["k10"]
            self.k12 = parameters["k12"]
            self.b12 = parameters["b12"]
            self.k21 = parameters["k21"]
            self.alpha0 = parameters["alpha0"]
            self.alpha1 = parameters["alpha1"]
            self.alpha2 = parameters["alpha2"]
            self.ktrans = parameters["ktrans"]
            self.delta = parameters["delta"]
        else:
            (
                self.r1,
                self.r2,
                self.k01,
                self.k10,
                self.k12,
                self.b12,
                self.k21,
                self.alpha0,
                self.alpha1,
                self.alpha2,
                self.ktrans,
                self.delta
            ) = parameters

        def propensity_t(t, out):
            out[2] = self.k12 + self.b12*np.exp(-self.r1*t)*(1.0-np.exp(-self.r2*t))

        def propensity_x(reaction, X, out):
            if reaction == 0:
                out[:] = self.k01*X[:,0]
            elif reaction == 1:
                out[:] = self.k10*X[:,1]
            elif reaction == 2:
                out[:] = X[:,1]
            elif reaction == 3:
                out[:] = self.k21*X[:,2]
            elif reaction == 4:
                out[:] = self.alpha0*X[:,0] + self.alpha1*X[:,1] + self.alpha2*X[:,2]
            elif reaction == 5:
                out[:] = self.ktrans*X[:, 3]
            elif reaction == 6:
                out[:] = self.delta*X[:, 4]

        self.propensity_t = propensity_t
        self.propensity_x = propensity_x

        def dpropt(parameter_idx: int, t: float, out: np.ndarray)->None:
            out[:] = 0.0
            if parameter_idx == 0:
                out[2] = -self.b12*t*np.exp(-self.r1*t)*(1.0-np.exp(-self.r2*t))
            elif parameter_idx == 1:
                out[2] = -self.b12*t*np.exp(-self.r1*t - self.r2*t)
            elif parameter_idx == 4:
                out[2] = 1.0
            elif parameter_idx == 5:
                out[2] = np.exp(-self.r1*t)*(1.0-np.exp(-self.r2*t))

        self.dpropt_sparsity = [[2], [2], [], [], [2], [2], [], [], [], [], [], []]

        def dpropx_dr1(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_dr2(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_dk01(reaction, X, out):
            if reaction == 0:
                out[:] = X[:, 0]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk10(reaction, X, out):
            if reaction == 1:
                out[:] = X[:, 1]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk12(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_db12(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_dk21(reaction, X, out):
            if reaction == 3:
                out[:] = X[:, 2]
            else:
                out[:] = 0.0
            return None

        def dpropx_da0(reaction, X, out):
            if reaction == 4:
                out[:] = X[:, 0]
            else:
                out[:] = 0.0
            return None

        def dpropx_da1(reaction, X, out):
            if reaction == 4:
                out[:] = X[:, 1]
            else:
                out[:] = 0.0
            return None

        def dpropx_da2(reaction, X, out):
            if reaction == 4:
                out[:] = X[:, 2]
            else:
                out[:] = 0.0
            return None

        def dpropx_dtrans(reaction, X, out):
            if reaction == 5:
                out[:] = X[:, 3]
            return None

        def dpropx_ddelta(reaction, X, out):
            if reaction == 6:
                out[:] = X[:, 4]
            return None

        self.dpropx_list = [
            dpropx_dr1,
            dpropx_dr2,
            dpropx_dk01,
            dpropx_dk10,
            dpropx_dk12,
            dpropx_db12,
            dpropx_dk21,
            dpropx_da0,
            dpropx_da1,
            dpropx_da2,
            dpropx_dtrans,
            dpropx_ddelta
        ]

        self.dpropx_sparsity = \
            [ [], [], [0], [1], [], [], [3], [4], [4], [4], [5], [6]]

        def dpropx(parameter_idx: int, reaction: int, x: np.ndarray, out: np.ndarray)->None:
            return self.dpropx_list[parameter_idx](reaction, x, out)

        self.dpropt = dpropt
        self.dpropx = dpropx