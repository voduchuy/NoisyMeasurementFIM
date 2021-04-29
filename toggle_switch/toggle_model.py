import numpy as np
from typing import Union, Dict

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "by": 6.8e-5,
    "bx": 2.2e-3,
    "ky": 1.6e-2,
    "kx": 1.7e-2,
    "axy": 6.1e-3,
    "ayx": 2.6e-3,
    "nxy": 2.1,
    "nyx": 3,
    "gammax": 3.8e-4,
    "gammay": 3.8e-4,
}


class BurstingGeneModel:
    X0 = [[0, 0]]
    P0 = [1.0]
    S0 = [0.0]
    init_bounds = np.array([20, 20])

    stoichMatrix = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def __init__(self, parameters: Union[np.ndarray, dict] = _DEFAULT_PARAMETERS):
        if type(parameters) == dict:
            self.by = parameters["by"]
            self.bx = parameters["bx"]
            self.ky = parameters["ky"]
            self.kx = parameters["kx"]
            self.axy = parameters["axy"]
            self.ayx = parameters["ayx"]
            self.nxy = parameters["nxy"]
            self.nyx = parameters["nyx"]
            self.gammax = parameters["gammax"]
            self.gammay = parameters["gammay"]
        else:
            (
                self.by,
                self.bx,
                selflky,
                self.kx,
                self.axy,
                self.ayx,
                self.nxy,
                self.nyx,
                self.gammax,
                self.gammay,
            ) = parameters

        # NEED TO CHANGE THIS
        # self.dprop_sparsity = np.eye(4, dtype=np.intc)

        # NEED TO CHANGE THIS
        def dprop_t_factory(i: int):
            def dprop_t(t, out):
                out[:] = 0.0
                out[i] = 1.0

            return dprop_t

        self.dprop_t_list = []
        for i in range(0, 4):
            self.dprop_t_list.append(dprop_t_factory(i))

        def propensity_t(t, out):
            out[0] = self.k01
            out[1] = self.k10
            out[2] = self.alpha
            out[3] = self.gamma
            return None

        def propensity_x(reaction, X, out):
            if reaction == 0:
                out[:] = X[:, 0]
                return None
            if reaction == 1:
                out[:] = X[:, 1]
                return None
            if reaction == 2:
                out[:] = X[:, 1]
                return None
            if reaction == 3:
                out[:] = X[:, 2]
                return None

        self.propensity_t = propensity_t
        self.propensity_x = propensity_x
