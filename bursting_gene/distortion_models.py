import numpy as np
from numpy.random import default_rng

RNG = default_rng()
#%%
class DistortionModel:
    def __init__(self):
        pass

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        """
        Generate a vector of conditional probabilities.

        Parameters
        ----------
        x :
            the true mRNA copy number to condition on.
        y :
            vector of observed values to evaluate the probabilities at.

        Returns
        -------
        p:
            vector of conditional probabilities of y given x.
        """
        pass

    def getDenseMatrix(self, xrange: np.ndarray, yrange: np.ndarray) -> np.ndarray:
        """
        Generate a dense matrix representation of the distortion operator.

        Parameters
        ----------
        xrange :
            Range of values for the latent mRNA copy number.

        yrange :
            Range of values for the observed mRNA copy number/intensity.

        Returns
        -------
        out:
            2-D array for the dense matrix with shape=(len(yrange), len(xrange)).
        """
        nx = len(xrange)
        ny = len(yrange)
        M = np.zeros((ny, nx))
        for j, x in enumerate(xrange):
            M[:, j] = self.getConditionalProbabilities(x, yrange)
        return M


#%% Binomial model
from scipy.stats import binom


class BinomialDistortionModel(DistortionModel):
    def __init__(self, detectionRate: float = 0.5):
        super().__init__()
        self.detectionRate = detectionRate

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return binom.pmf(y, x, self.detectionRate)

#%% Additive Poisson Noise
from scipy.stats import poisson

class AdditivePoissonDistortionModel(DistortionModel):
    def __init__(self, poissonRate: float = 10.0):
        super().__init__()
        self.rate = poissonRate

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return poisson.pmf(y - x, self.rate)
#%% Flow-cytometry measurement
class FlowCytometryModel(DistortionModel):
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

    def sampleObservations(self, x: np.ndarray, rng=RNG) -> np.ndarray:
        ans = rng.normal(
            loc=self.mu_probe * x + self.mu_bg,
            scale=np.sqrt(x * self.sigma_probe ** 2.0 + self.sigma_bg ** 2.0),
        )
        return ans
