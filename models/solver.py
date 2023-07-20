import numpy as np
from util.possible_states import setup_possible_states_array

class Solver(object):
    def __init__(self, env, debug=True):
        self.env = env
        self.gamma = 1
        self.max_delta = 0.05
        self.eps = 0.15
        self.value = None
        self.policy = None
        self.debug = debug
        self.possible_states_array = setup_possible_states_array(env)
        self.reset()

    def reset(self):
        #self.value = np.zeros(self.env.observation_space.nvec)
        self.value = np.zeros((self.env.booking_time, self.env.flight_capacity))
        self.policy = np.zeros((self.env.booking_time, self.env.flight_capacity))
        # Zero Starting Policy
        #self.policy = np.zeros(self.env.observation_space.nvec)
        # Random Starting Policy
        #self.policy = np.random.choice(np.arange(int(self.env.action_space_max)), size=self.env.observation_space.nvec)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()