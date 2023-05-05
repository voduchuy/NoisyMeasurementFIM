# NoisyMeasurementFIM
Code to reproduce numerical simulations and figures from the [preprint](https://doi.org/10.1101/2021.05.11.443611):

Huy D. Vo, Linda Forero, Luis Aguilera, Brian Munsky. _Analysis and design of single-cell experiments to harvest fluctuation information while rejecting measurement noise_


## Dependencies
For computing the FSP sensitivity solutions and MLE validations:
- CME solvers: [PACMENSL](https://github.com/voduchuy/pacmensl) and [PyPACMENSL](https://github.com/voduchuy/pypacmensl) (only for the FSP solutions).
- PyGMO 2 (for MLE validations only).

For plotting the numerial results:
- Python 3X.
- NumPy 1.22.
- Matplotlib.
- Pandas.
- A working installation of [LaTex](https://www.latex-project.org/get/).

The dependency PACMENSL is a C++ library for parallel FSP solution of the chemical master equation. Follow the steps in the README file of that repository to compile the library. We note that it is only needed if the reader wishes to repeat the computation scripts. If the reader only wants to inspect the numerical results and reproduce the figures in the manuscript, only Python packages are needed. We provide convenient scripts and configuration files to set up a Python environment quickly for plotting the results (see below).

## Setting up environment to reproduce figures
### Downloading the numerical results
Readers who do not wish to repeat the time-consuming calculations (which requires installing the C++ dependencies and may require cluster access) may simply [download](https://zenodo.org/record/7880494#.ZFOUE-zMJhE) the numerical results from Zenodo. A convenient Python script (`download_results.py`) is provided in the root folder that will download the Zip file, unpack, and populate the subfolders of the repository with `.npz` and `.png` files needed for plotting.

### Setting up Python environment
This step requires that [Anaconda](https://www.anaconda.com/) is installed on the reader's computer. An environment file `environment.yml` is provided in the root folder of this repository. Readers can use Anaconda to set up the Python environment for executing the notebooks by running the following command in Terminal:
```zsh
conda env create -f environment.yml
```

This will create an environment named `fimhuyvo2022` in the user's default environment path. The Python packages needed for executing the provided notebooks will be installed in this environment. Users can then activate the environment
```zsh
conda activate fimhuyvo2022
```
Provided that the numerical outputs have been downloaded and the files are copied to the correct locations (see above), the Jupyter notebooks can be executed.

### Executing the Jupyter notebooks
After the environment has been created and activated, user can run `jupyter notebook` from the Terminal, change dir to one of the subfolders `bursting_gene`, `toggle_switch`, `yeast`. A single Jupyter notebook is provided in each subfolder that produce all figures related to the corresponding CME (telegraph, 2-D toggle switch, or compartmental MAPK-activated gene expression). 

### Cleaning up
The Anaconda environment can be removed with
```
conda env remove -n fimhuyvo2022
```
