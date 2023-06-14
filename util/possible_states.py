import numpy as np
from itertools import product

def setup_possible_states_array(env):
        max_values = env.observation_space.nvec
        ranges = [range(max_val) for max_val in max_values]
        return np.array(list(product(*ranges)))