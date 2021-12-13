import numpy as np

# Number of sampling times in the experiment design. These times are equally spaced and the gap is a variable to be
# optimized for maximal information
num_sampling_times = 5


def compute_combined_fim(fim_array, dt, num_times):
    t_idx = dt * np.linspace(1, num_times, num_times, dtype=int)
    fim = fim_array[t_idx[0], :, :]
    for i in range(1, len(t_idx)):
        fim += fim_array[t_idx[i], :, :]
    return fim
