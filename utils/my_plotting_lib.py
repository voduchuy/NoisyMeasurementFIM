import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_conf_ellipse(fim: np.ndarray,
                      num_sigma: int,
                      ax: plt.Axes,
                      par_idx: [int],
                      theta: np.ndarray,
                      color='red', label='my_ellipse'):
    covmat = np.linalg.inv(fim)
    [eigvals, eigvecs] = np.linalg.eig(covmat[np.ix_([par_idx[0], par_idx[1]], [par_idx[0], par_idx[1]])])

    indx = np.argsort(eigvals)
    indx = np.flip(indx)
    eigvals = eigvals[indx]
    eigvecs = eigvecs[:, indx]

    mu0 = np.log10(theta[par_idx[0]])
    mu1 = np.log10(theta[par_idx[1]])
    sigma0 = np.sqrt(eigvals[0])
    sigma1 = np.sqrt(eigvals[1])
    a = num_sigma * sigma0
    b = num_sigma * sigma1

    ax.axvline(mu0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(mu1, color='k', linestyle='--', alpha=0.5)

    phi = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    if phi < 0:
        phi = phi + 2 * np.pi

    rot_matrix = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
    ])

    phi_grid = np.linspace(0, 2 * np.pi, 100)
    ellipse_x_r = a * np.cos(phi_grid)
    ellipse_y_r = b * np.sin(phi_grid)

    r_ellipse = np.array(rot_matrix @ [ellipse_x_r, ellipse_y_r])

    ax.plot(r_ellipse[0, :] + mu0, r_ellipse[1, :] + mu1, color=color, label=label)

    # Plot the major ax
    ax.plot([mu0 - eigvecs[0, 0] * a, mu0 + eigvecs[0, 0] * a], [mu1 - eigvecs[1, 0] * a, mu1 + eigvecs[1, 0] * a],
            color=color,
            linestyle='--')

    return 0


def plot_barcodes(fim_mats: [np.ndarray],
                  labels: [str],
                  colors: [str],
                  ax: plt.Axes):
    # The uncertainties in eigendirections of different experiments
    for i in range(0, len(fim_mats)):
        [eigval, eigvec] = np.linalg.eig(fim_mats[i])
        uncertainties = 1 / np.sqrt(eigval)

        ax.hlines(uncertainties, xmin=(2 * i + 1) - 0.75, xmax=(2 * i + 1) + 0.75, color=colors[i], linewidth=2)

    ax.set_yscale('log')
    ax.set_xticks(2 * np.arange(0, len(fim_mats)) + 1)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Uncertainty')

    return 0


from string import ascii_uppercase


def label_axes(axs: [plt.Axes]):
    for i, ax in enumerate(list(axs.flatten())):
        ax.set_title(ascii_uppercase[i])