import numpy as np
from typing import Union, Dict

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "k01": 0.05,
    "k10": 0.015,
    "alpha": 5.0,
    "gamma": 0.05,
}


class BurstingGeneModel:
    X0 = [[1, 0, 0]]
    P0 = [1.0]
    S0 = [0.0]
    init_bounds = np.array([1, 1, 20])

    stoichMatrix = [[-1, 1, 0], [1, -1, 0], [0, 0, 1], [0, 0, -1]]

    def __init__(self, parameters: Union[np.ndarray, dict] = _DEFAULT_PARAMETERS):
        if type(parameters) == dict:
            self.k01 = parameters["k01"]
            self.k10 = parameters["k10"]
            self.alpha = parameters["alpha"]
            self.gamma = parameters["gamma"]
        else:
            self.k01, self.k10, self.alpha, self.gamma = parameters

        self.dprop_sparsity = np.eye(4, dtype=np.intc)

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
