import numpy as np
from util.possible_states import setup_possible_states_array, setup_possible_events_array
from scipy.stats import poisson
from duopoly_environment import DuopolyEnvironment

class Solver(object):
    def __init__(self, env, estimator=None, debug=False):
        self.env = env
        if isinstance(self.env, DuopolyEnvironment):
            self.duopol = True
        else:
            self.duopol = False
        self.estimator = estimator
        self.gamma = 1
        self.max_delta = 0.005
        self.eps = 0.15
        self.value = None
        self.policy = None
        self.debug = debug
        self.possible_states_array = setup_possible_states_array(env)
        self.possible_events_array = setup_possible_events_array(env)
        self.reset()

    def reset(self):
        self.value = np.zeros(self.env.observation_space.nvec, dtype=int)
        # Zero Starting Policy
        self.policy = np.zeros(self.env.observation_space.nvec, dtype=int)
        # Random Starting Policy
        # self.policy = np.random.choice(np.arange(int(self.env.action_space_max)), size=self.env.observation_space.nvec)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()
    
    def event_p(self, i, a, s):
        if self.estimator:
            estimated_function = self.estimator.estimate_function
            if self.duopol:
                return poisson.pmf(i[1], mu=estimated_function(x=a, t=s[0], a_c=s[3], i_c=i[2]))
            else:
                return poisson.pmf(i[1], mu=estimated_function(x=a, t=s[0]))
        else:
            return self.env.get_event_p(i, a, s)