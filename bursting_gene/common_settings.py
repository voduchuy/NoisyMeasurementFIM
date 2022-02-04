import numpy as np

# Number of sampling times in the experiment design. These times are equally spaced and the gap is a variable to be
# optimized for maximal information
NUM_SAMPLING_TIMES = 5


def computeCombinedFim(fim_array, dt, num_times):
    t_idx = dt * np.arange(1, num_times+1)
    fim = fim_array[t_idx[0], :, :]
    for i in range(1, len(t_idx)):
        fim += fim_array[t_idx[i], :, :]
    return fim

def computeFimEntry(a, b, p, C=None):
    xmax = len(p) - 1
    Cp = C[:, 0 : xmax + 1] @ p if C is not None else p
    Ca = C[:, 0 : xmax + 1] @ a if C is not None  else a
    Cb = C[:, 0 : xmax + 1] @ b if C is not None  else b
    return np.sum(Ca * Cb / np.maximum(1.0e-16, Cp))

def computeSingleObservationFim(distributions, sensitivities, distortionMatrix=None):
    numTimes = len(distributions)
    numParameters = len(sensitivities[0])

    fim = np.zeros((numTimes, numParameters, numParameters))
    for itime in range(numTimes):
        for ip in range(0, numParameters):
            for jp in range(0, ip+1):
                fim[itime, ip, jp] = computeFimEntry(
                    sensitivities[itime][ip],
                    sensitivities[itime][jp],
                    distributions[itime],
                    distortionMatrix,
                )
        for ip in range(0, numParameters):
            for jp in range(ip + 1, numParameters):
                fim[itime, ip, jp] = fim[itime, jp, ip]
    return fim