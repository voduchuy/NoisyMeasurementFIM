import numpy as np
from typing import Union, Dict

# Parameters for STL1 from PNAS paper
_DEFAULT_PARAMETERS: Dict[str, float] = {
    "k_01": 2.6e-3,
    "k_10a": 1.9e01,
    "k_10b": 3.2e04,
    "k_12": 7.63e-3,
    "k_21": 1.2e-2,
    "k_23": 4e-3,
    "k_32": 3.1e-3,
    "alpha_0": 5.9e-4,
    "alpha_1": 1.7e-1,
    "alpha_2": 1.0,
    "alpha_3": 3e-2,
    "k_trans": 2.6e-1,
    "gamma_nuc": 2.2e-6,
    "gamma_cyt": 8.3e-3,
}


def fsp_constraints(X, out):
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 2]
    out[:, 3] = X[:, 3]
    out[:, 4] = X[:, 4]
    out[:, 5] = X[:, 5]
    out[:, 6] = X[:, 4] + X[:, 5]


class YeastModel:
    X0 = [[1, 0, 0, 0, 0, 0]]
    P0 = [1.0]
    S0 = [0.0]
    NUM_PARAMETERS = 14

    init_bounds = np.array([1, 1, 1, 1, 10, 10, 10])

    stoichMatrix = [
        [-1, 1, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0],
        [0, 1, -1, 0, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, -1],
    ]

    # Parameters for the HOG1p signal
    r1 = 6.1e-3
    r2 = 6.9e-3
    eta = 5.9
    Ahog = 9.3e9
    Mhog = 2.2e-2

    def __init__(self, parameters: Union[np.ndarray, dict] = _DEFAULT_PARAMETERS, osmotic=True):
        if type(parameters) == dict:
            self.k_01 = parameters["k_01"]
            self.k_10a = parameters["k_10a"]
            self.k_10b = parameters["k_10b"]
            self.k_12 = parameters["k_12"]
            self.k_21 = parameters["k_21"]
            self.k_23 = parameters["k_23"]
            self.k_32 = parameters["k_32"]
            self.alpha_0 = parameters["alpha_0"]
            self.alpha_1 = parameters["alpha_1"]
            self.alpha_2 = parameters["alpha_2"]
            self.alpha_3 = parameters["alpha_3"]
            self.k_trans = parameters["k_trans"]
            self.gamma_nuc = parameters["gamma_nuc"]
            self.gamma_cyt = parameters["gamma_cyt"]
        else:
            (
                self.k_01,
                self.k_10a,
                self.k_10b,
                self.k_12,
                self.k_21,
                self.k_23,
                self.k_32,
                self.alpha_0,
                self.alpha_1,
                self.alpha_2,
                self.alpha_3,
                self.k_trans,
                self.gamma_nuc,
                self.gamma_cyt,
            ) = parameters

        def propensity_t_no_osmotic(t, out):
            out[1] = self.k_10a

        def propensity_t_with_osmotic(t, out):
            u = (1.0 - np.exp(-self.r1 * t)) * np.exp(-self.r2 * t)
            signal = self.Ahog * (u / (1.0 + u / self.Mhog)) ** self.eta
            out[1] = max([0.0, self.k_10a - self.k_10b * signal])

        def propensity_x(reaction, X, out):
            if reaction == 0:
                out[:] = self.k_01 * X[:, 0]
            elif reaction == 1:
                out[:] = X[:, 1]
            elif reaction == 2:
                out[:] = self.k_12 * X[:, 1]
            elif reaction == 3:
                out[:] = self.k_21 * X[:, 2]
            elif reaction == 4:
                out[:] = self.k_23 * X[:, 2]
            elif reaction == 5:
                out[:] = self.k_32 * X[:, 3]
            elif reaction == 6:
                out[:] = (
                      self.alpha_0 * X[:, 0]
                    + self.alpha_1 * X[:, 1]
                    + self.alpha_2 * X[:, 2]
                    + self.alpha_3 * X[:, 3]
                )
            elif reaction == 7:
                out[:] = self.k_trans * X[:, 4]
            elif reaction == 8:
                out[:] = self.gamma_nuc * X[:, 4]
            elif reaction == 9:
                out[:] = self.gamma_cyt * X[:, 5]

        self.propensity_t = propensity_t_no_osmotic if not osmotic else propensity_t_with_osmotic
        self.propensity_x = propensity_x

        def dpropt_no_osmotic(parameter_idx: int, t: float, out: np.ndarray) -> None:
            if parameter_idx == 1:
                out[1] = 1.0
            else:
                out[1] = 0.0

        def dpropt_with_osmotic(parameter_idx: int, t: float, out: np.ndarray) -> None:
            self.propensity_t(t, out)
            if parameter_idx == 1:
                out[1] = 1.0 if out[1] >= 0 else 0.0
            elif parameter_idx == 2:
                u = (1.0 - np.exp(-self.r1 * t)) * np.exp(-self.r2 * t)
                signal = self.Ahog * (u / (1.0 + u / self.Mhog)) ** self.eta
                out[1] = -1.0 * signal if out[1] >= 0.0 else 0.0

        self.dpropt_sparsity = [
            [],
            [1],
            [1],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ]

        def dpropx_dk_01(reaction, X, out):
            if reaction == 0:
                out[:] = X[:, 0]
            return None

        def dpropx_dk_10a(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_dk_10b(reaction, X, out):
            out[:] = 0.0
            return None

        def dpropx_dk_12(reaction, X, out):
            if reaction == 2:
                out[:] = X[:, 1]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk_21(reaction, X, out):
            if reaction == 3:
                out[:] = X[:, 2]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk_23(reaction, X, out):
            if reaction == 4:
                out[:] = X[:, 2]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk_32(reaction, X, out):
            if reaction == 5:
                out[:] = X[:, 3]
            else:
                out[:] = 0.0
            return None

        def dpropx_da_0(reaction, X, out):
            if reaction == 6:
                out[:] = X[:, 0]
            else:
                out[:] = 0.0
            return None

        def dpropx_da_1(reaction, X, out):
            if reaction == 6:
                out[:] = X[:, 1]
            else:
                out[:] = 0.0
            return None

        def dpropx_da_2(reaction, X, out):
            if reaction == 6:
                out[:] = X[:, 2]
            else:
                out[:] = 0.0
            return None

        def dpropx_da_3(reaction, X, out):
            if reaction == 6:
                out[:] = X[:, 3]
            else:
                out[:] = 0.0
            return None

        def dpropx_dk_trans(reaction, X, out):
            if reaction == 7:
                out[:] = X[:, 4]
            return None

        def dpropx_dgamma_nuc(reaction, X, out):
            if reaction == 8:
                out[:] = X[:, 4]
            return None

        def dpropx_dgamma_cyt(reaction, X, out):
            if reaction == 9:
                out[:] = X[:, 5]
            return None

        self.dpropx_list = [
            dpropx_dk_01,
            dpropx_dk_10a,
            dpropx_dk_10b,
            dpropx_dk_12,
            dpropx_dk_21,
            dpropx_dk_23,
            dpropx_dk_32,
            dpropx_da_0,
            dpropx_da_1,
            dpropx_da_2,
            dpropx_da_3,
            dpropx_dk_trans,
            dpropx_dgamma_nuc,
            dpropx_dgamma_cyt,
        ]

        self.dpropx_sparsity = [
            [0],
            [],
            [],
            [2],
            [3],
            [4],
            [5],
            [6],
            [6],
            [6],
            [6],
            [7],
            [8],
            [9],
        ]

        def dpropx(
            parameter_idx: int, reaction: int, x: np.ndarray, out: np.ndarray
        ) -> None:
            return self.dpropx_list[parameter_idx](reaction, x, out)

        self.dpropt = dpropt_no_osmotic if not osmotic else dpropt_with_osmotic
        self.dpropx = dpropx
