import numpy as np
from numpy.random import Generator
from numpy.random import default_rng
from .base import DistortionModel

RNG = default_rng()

#%% Binomial model
from scipy.stats import binom

class BinomialDistortionModel(DistortionModel):
    def __init__(self, detectionRate: float = 0.5):
        super().__init__()
        self.detectionRate = detectionRate

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return binom.pmf(y, x, self.detectionRate)

#%% Binomial model with state-dependent detection rate
class BinomialVaryingDetectionRate(DistortionModel):
    def __init__(self):
        super().__init__()

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        detectionRates = 1.0/(1.0 + 0.01*x)
        return binom.pmf(y, x, detectionRates)

    def distort(self, x: np.ndarray, rng: Generator = default_rng())->np.ndarray:
        detectionRates = 1.0/(1.0 + 0.01*x)
        return rng.binomial(n=x, p=detectionRates)
#%% Binning with uniform bin widths
class UniformBinning(DistortionModel):
    def __init__(self, width=10):
        super().__init__()
        self.width = width

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return np.array((y*self.width <= x) & ((y+1)*self.width > x), dtype=np.double)
#%% Additive Poisson Noise
from scipy.stats import poisson

class AdditivePoissonDistortionModel(DistortionModel):
    def __init__(self, poissonRate: float = 10.0):
        super().__init__()
        self.rate = poissonRate

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return poisson.pmf(y - x, self.rate)
#%% Integrated intensity measurement
class IntegratedIntensityModel(DistortionModel):
    def __init__(
        self,
        mu_probe: float = 25,
        sigma_probe: float = np.sqrt(25),
        mu_bg: float = 200,
        sigma_bg: float = 400,
    ):
        self.mu_probe = mu_probe
        self.sigma_probe = sigma_probe
        self.mu_bg = mu_bg
        self.sigma_bg = sigma_bg

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return np.exp(
            -((y - x * self.mu_probe - self.mu_bg) ** 2.0)
            / (2.0 * (x * self.sigma_probe ** 2.0 + self.sigma_bg ** 2.0))
        ) / np.sqrt(2 * np.pi * (x * self.sigma_probe ** 2.0 + self.sigma_bg ** 2.0))

    def distort(self, x: np.ndarray, rng=RNG) -> np.ndarray:
        ans = rng.normal(
            loc=self.mu_probe * x + self.mu_bg,
            scale=np.sqrt(x * self.sigma_probe ** 2.0 + self.sigma_bg ** 2.0),
        )
        return ans