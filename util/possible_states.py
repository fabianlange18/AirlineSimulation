import numpy as np
from itertools import product

def setup_possible_states_array(env):
        max_values = env.observation_space.nvec
        # ranges = [range(max_values[0])]
        # ranges.append(*[range(max_val - 1, -1, -1) for max_val in max_values[1:]])
        ranges = [range(max_val) for max_val in max_values]
        return np.array(list(product(*ranges)))