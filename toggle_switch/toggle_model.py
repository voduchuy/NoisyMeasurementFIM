import numpy as np
from typing import Union, Dict

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "bx": 2.2e-3,
    "by": 6.8e-5,
    "kx": 1.7e-2,
    "ky": 1.6e-2,
    "ayx": 2.6e-3,
    "axy": 6.1e-3,
    "nyx": 3,
    "nxy": 2.1,
    "gammax": 3.8e-4,
    "gammay": 3.8e-4,
}


class ToggleSwitchModel:
    X0 = [[0, 0]]
    P0 = [1.0]
    S0 = [0.0]
    NUM_PARAMETERS = 10
    init_bounds = np.array([20, 20])

    stoichMatrix = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def __init__(self, parameters: Union[np.ndarray, dict] = _DEFAULT_PARAMETERS):
        if type(parameters) == dict:
            self.bx = parameters["bx"]
            self.by = parameters["by"]
            self.kx = parameters["kx"]
            self.ky = parameters["ky"]
            self.ayx = parameters["ayx"]
            self.axy = parameters["axy"]
            self.nyx = parameters["nyx"]
            self.nxy = parameters["nxy"]
            self.gammax = parameters["gammax"]
            self.gammay = parameters["gammay"]
        else:
            (
                self.bx,
                self.by,
                self.kx,
                self.ky,
                self.ayx,
                self.axy,
                self.nyx,
                self.nxy,
                self.gammax,
                self.gammay,
            ) = parameters

        def propensity_t(t, out):
            out[:] = 1.0
            return None

        def propensity_x(reaction, X, out):
            if reaction == 0:
                out[:] = self.bx + self.kx / (1 + self.ayx * X[:, 1] ** self.nyx)
                return None
            if reaction == 1:
                out[:] = self.gammax * X[:, 0]
                return None
            if reaction == 2:
                out[:] = self.by + self.ky / (1 + self.axy * X[:, 0] ** self.nxy)
                return None
            if reaction == 3:
                out[:] = self.gammay * X[:, 1]
                return None

        self.propensity_t = propensity_t
        self.propensity_x = propensity_x

        # NEED TO CHANGE THIS
        self.dprop_sparsity = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.intc)

        def dprop_t_factory(i: int):
            def dprop_t(t, out):
                out[:] = 0.0

            return dprop_t

        self.dprop_t_list = []
        for i in range(0, 10):
            self.dprop_t_list.append(dprop_t_factory(i))

        def dpropx_dbx(reaction, X, out):
            if reaction == 0:
                out[:] = 1.0
            else:
                out[:] = 0.0
            return None

        def dpropx_dby(reaction, X, out):
            if reaction == 2:
                out[:] = 1.0
            else:
                out[:] = 0.0
            return None

        def dpropx_dkx(reaction, X, out):
            if reaction == 0:
                out[:] = 1.0 / (1 + self.ayx * X[:, 1] ** self.nyx)
            else:
                out[:] = 0.0
            return None

        def dpropx_dky(reaction, X, out):
            if reaction == 2:
                out[:] = 1.0 / (1 + self.axy * X[:, 0] ** self.nxy)
            else:
                out[:] = 0.0
            return None

        def dpropx_dayx(reaction, X, out):
            if reaction == 0:
                out[:] = (
                    -(X[:, 1] ** self.nyx) / (1 + self.ayx * X[:, 1] ** self.nyx) ** 2.0
                )
            else:
                out[:] = 0.0
            return None

        def dpropx_daxy(reaction, X, out):
            if reaction == 2:
                out[:] = (
                    -(X[:, 0] ** self.nxy) / (1 + self.axy * X[:, 0] ** self.nxy) ** 2.0
                )
            else:
                out[:] = 0.0
            return None

        def dpropx_dnyx(reaction, X, out):
            if reaction == 0:
                out[:] = (
                    -(self.nyx * self.ayx * X[:, 1] ** (self.nyx - 1.0))
                    / (1 + self.ayx * X[:, 1] ** self.nyx) ** 2.0
                )
            else:
                out[:] = 0.0
            return None

        def dpropx_dnxy(reaction, X, out):
            if reaction == 2:
                out[:] = (
                    -(self.nxy * X[:, 0] ** (self.nxy - 1.0))
                    / (1 + self.axy * X[:, 0] ** self.nxy) ** 2.0
                )
            else:
                out[:] = 0.0
            return None

        def dpropx_dgx(reaction, X, out):
            if reaction == 1:
                out[:] = X[:, 0]
            return None

        def dpropx_dgy(reaction, X, out):
            if reaction == 3:
                out[:] = X[:, 1]
            return None

        self.dprop_x_list = [
            dpropx_dbx,
            dpropx_dby,
            dpropx_dkx,
            dpropx_dky,
            dpropx_dayx,
            dpropx_daxy,
            dpropx_dnyx,
            dpropx_dnxy,
            dpropx_dgx,
            dpropx_dgy
        ]
