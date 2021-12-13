import numpy as np
from typing import Union, NoReturn


def compute_fim_functional(fims: np.ndarray, criteria: str= "d") -> np.ndarray:
    """
    y = compute_fim_functional(fims, criteria)
    
    Compute optimality criteria for a collection (array) of Fisher Information Matrices. These criteria could
    either be the determinant (D-Optimality), the smallest eigenvalue (E-Optimality), or the trace
    of the matrix (A-optimality).
    
    Parameters
    ----------
    fims : array with number of dimensions >= 2
        numpy array that stores the Fisher Information Matrices. We assume the last two dimensions to represent
        the parameters. 
        
    criteria : str
        either "d", "e", or "a". Default: "d".

    Returns
    -------
    y: array 
        values of the optimality criteria, satisfying the relation
        y.ndim = fims.ndim - 2.
    """

    criteria_func = {
        'd': lambda x: np.linalg.det(x),
        'a': lambda x: np.trace(x),
        'e': lambda x: np.linalg.eigvalsh(x)[0]
    }
    func = criteria_func[criteria]

    if len(fims.shape) < 2:
        raise ValueError("Input FIMS must have at least two dimensions.")

    if len(fims.shape) == 2:
        return func(fims)

    y = np.zeros(fims.shape[:-2])
    y_view = np.reshape(y, (-1, 1))
    fims_view = np.reshape(fims, (np.prod(fims.shape[:-2]), fims.shape[-2], fims.shape[-1]))

    for i in range(y_view.shape[0]):
        y_view[i] = func(fims_view[i, :, :])

    return y


def log_transform(fims: np.ndarray, theta: np.ndarray, inplace=True)->Union[np.ndarray, NoReturn]:
    num_parameter = len(theta)
    if num_parameter != fims.shape[-1]:
        raise ValueError("Dimensions of parameter vector and the Fisher Information Matrices do not match.")
    if inplace:
        out = fims
    else:
        out = np.copy(fims)
    for i in range(0, num_parameter):
        for j in range(0, num_parameter):
            out[..., i, j] *= theta[i]*theta[j]*np.log(10)*np.log(10)
    return out



if __name__ == "__main__":
    mats = np.ones((2,2,2,2))
    aopt = compute_fim_functional(mats, "a")
    dopt = compute_fim_functional(mats, "d")
    eopt = compute_fim_functional(mats, "e")
    for i in range(2):
        for j in range(2):
            assert(aopt[i,j] == 2)
            assert(dopt[i,j] == 0)
            assert(eopt[i,j] == 0)
