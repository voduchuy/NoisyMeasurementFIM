import numpy as np 
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
