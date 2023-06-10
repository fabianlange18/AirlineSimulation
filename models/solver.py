import numpy as np
from itertools import product

class Solver(object):
    def __init__(self, env, debug=True):
        self.env = env
        self.gamma = 1
        self.max_delta = 0.05
        self.eps = 0.15
        self.value = None
        self.policy = None
        self.debug = debug
        self.possible_states_array = self.setup_possible_states_array()
        self.reset()

    def reset(self):
        self.value = np.zeros(self.env.observation_space.nvec)
        # Zero Starting Policy
        self.policy = np.zeros(self.env.observation_space.nvec)
        # Random Starting Policy
        # self.policy = np.random.choice(np.arange(int(self.env.action_space_max)), size=self.env.observation_space.nvec)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def setup_possible_states_array(self):
        max_values = self.env.observation_space.nvec
        ranges = [range(max_values[0])]
        ranges.append(*[range(max_val - 1, -1, -1) for max_val in max_values[1:]])
        return np.array(list(product(*ranges)))