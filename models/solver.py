import numpy as np


class Solver(object):
    def __init__(self, env, debug=True):
        self.env = env
        self.gamma = 0.95
        self.eps = 0.15
        self.value = None
        self.policy = None
        # self.debug = debug
        self.reset()

    def reset(self):
        self.value = np.zeros(self.env.observation_space.nvec)
        self.policy = np.random.choice(np.arange(int(self.env.action_space.n)), size=self.env.observation_space.nvec)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()
