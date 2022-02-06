from .base import DistortionModel
import numpy as np


class DoubleCell(DistortionModel):
    def __init__(self, rho: float = 0.01):
        self.rho = rho
        """The probability the mRNA count in a cell is added to that of another independent cell"""

    def transformDistribution(self, pvec: np.ndarray) -> np.ndarray:
        return self.rho * np.convolve(pvec, pvec) + (1.0 - self.rho) * np.pad(
            pvec, pad_width=(0, len(pvec) - 1), mode="constant", constant_values=0.0
        )

    def transformSensitivity(self, pvec: np.ndarray, svec: np.ndarray) -> np.ndarray:
        return self.rho * np.convolve(svec, pvec) + self.rho*np.convolve(pvec, svec) + (1.0 - self.rho) * np.pad(
            svec, pad_width=(0, len(pvec) - 1), mode="constant", constant_values=0.0
        )
