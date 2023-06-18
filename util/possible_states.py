import numpy as np
from itertools import product

def setup_possible_states_array(env):
        ranges = [range(max_val) for max_val in env.observation_space.nvec]
        return np.array(list(product(*ranges)))