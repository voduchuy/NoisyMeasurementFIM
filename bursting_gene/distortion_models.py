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


#%% Low-resolution image noise
from scipy.special import comb


class LowResolutionModel(DistortionModel):
    # We consider spot miscounting due to low image resolution. Let $S$ be the size of the cell in terms of the pixels it occupies. Ignore for now the variability in cell sizes. Bright spots may clutter within a pixel, making the image looks like there is only one spot when there should actually be several of them.
    #
    # Given cell size S and n mRNA molecules, the recorded number of spots is given by the formula
    #
    # $$
    # P(Y = j | S, n)
    # =
    # \frac
    # {\binom{S}{j}\binom{n-1}{j-1}}
    # {\binom{S + n - 1}{n}}
    # $$
    #
    # This comes from a combinatoric problem:
    # Given $S$ boxes and $n$ apples,
    # how many ways are there to distribute these apples into these boxes such that there are exactly $j$ non-empty boxes.
    def __init__(self, numPixels: int = 100):
        """

        Parameters
        ----------
        numPixels : int
            number of pixels a single cell occupies.
        """
        self.numPixels = numPixels

    def getConditionalProbabilities(self, x: int, y: np.ndarray) -> np.ndarray:
        return (
            comb(self.numPixels, y)
            * comb(x - 1, y - 1)
            / comb(self.numPixels + x - 1, x)
        )

    def sampleObservations(self, x: np.ndarray, rng=RNG) -> np.ndarray:
        ans = np.zeros(x.shape[0], dtype=int)
        for i, rnaCount in enumerate(x):
            tmp = rng.choice(self.numPixels, size=rnaCount, replace=True)
            ans[i] = len(np.unique(tmp))
        return ans


#%% Flow-cytometry measurement
class FlowCytometryModel(DistortionModel):
    def __init__(
        self,
        mu_probe: float = 300,
        sigma_probe: float = 300,
        mu_bg: float = 100,
        sigma_bg: float = 200,
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
