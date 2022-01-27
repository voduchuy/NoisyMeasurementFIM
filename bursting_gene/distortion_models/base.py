import numpy as np
from numpy.random import Generator, default_rng
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
        raise RuntimeError("Not implemented.")

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

    def distort(self, x: np.ndarray, rng: Generator=default_rng())->np.ndarray:
        """
        Apply distortion on an array of true mRNA copy numbers to get the samples of distorted measurements.

        Parameters
        ----------
        x: 1-D array
            Array of latent mRNA copy numbers.

        rng:
            Random number generator. Default to Numpy's default_rng().

        Returns
        -------
        out: 1-D array
            Sampled distorted measurements, out[i] is the distorted version of x[i].
        """
        raise RuntimeError("Not implemented.")

#%%
class ZeroDistortion(DistortionModel):
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
        return (y == x).astype("float")

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

    def distort(self, x: np.ndarray, rng: Generator=default_rng())->np.ndarray:
        """
        Apply distortion on an array of true mRNA copy numbers to get the samples of distorted measurements.

        Parameters
        ----------
        x: 1-D array
            Array of latent mRNA copy numbers.

        rng:
            Random number generator. Default to Numpy's default_rng().

        Returns
        -------
        out: 1-D array
            Sampled distorted measurements, out[i] is the distorted version of x[i].
        """
        return np.copy(x)