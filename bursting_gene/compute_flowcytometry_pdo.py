import numpy as np
from distortion_models import FlowCytometryModel

flowcyt_model = FlowCytometryModel()
xrange = np.arange(0, 200)
yrange = np.linspace(flowcyt_model.mu_bg - 3*flowcyt_model.sigma_bg, flowcyt_model.mu_probe*200 + 3*flowcyt_model.sigma_bg, 1000)
C = flowcyt_model.getDenseMatrix(xrange, yrange)

np.savez('results/distortion_matrix_flowcyt.npz', C=C, xrange=xrange, yrange=yrange)

