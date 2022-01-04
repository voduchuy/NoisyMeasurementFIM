import numpy as np
from distortion_models import SmFishDistortion

if __name__ == "__main__":
    distortionMatrices = {}
    concentrationLevels = [0.1, 1.0, 5.0, 10.0]
    for probeConcentration in concentrationLevels:
        distortionMatrix = SmFishDistortion(probe_concentration=probeConcentration)
        x = np.arange(300)
        y = np.arange(400)
        C = distortionMatrix.getDenseMatrix(x, y)
        distortionMatrices[probeConcentration] = C
    np.savez("results/smfish_probe_distortions.npz", pdos = distortionMatrices, levels=concentrationLevels)
