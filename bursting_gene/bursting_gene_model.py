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

        self.dpropx_sparsity = [[0], [1], [2], [3]]
        self.dpropt_sparsity = None

        def propensity_t(t, out):
            out[:] = 0.0
            return None

        def propensity_x(reaction, X, out):
            if reaction == 0:
                out[:] = self.k01*X[:, 0]
                return None
            if reaction == 1:
                out[:] = self.k10*X[:, 1]
                return None
            if reaction == 2:
                out[:] = self.alpha*X[:, 1]
                return None
            if reaction == 3:
                out[:] = self.gamma*X[:, 2]
                return None

        def d_prop_x(par_idx, reaction, X, out):
            if par_idx == 0:
                if reaction == 0:
                    out[:] = X[:, 0]
            elif par_idx == 1:
                if reaction == 1:
                    out[:] = X[:, 1]
            elif par_idx == 2:
                if reaction == 2:
                    out[:] = X[:, 1]
            elif par_idx == 3:
                if reaction == 3:
                    out[:] = X[:, 2]
            else:
                raise ValueError("Invalid parameter index.")

        self.propensity_t = None
        self.propensity_x = propensity_x
        self.dpropensity_t = None
        self.dpropensity_x = d_prop_x



