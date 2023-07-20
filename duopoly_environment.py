import gym

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

import random
import numpy as np
from scipy.stats import multinomial
from scipy.special import softmax

from customers import Customers

class DuopolyEnvironment(gym.Env):

    def __init__(self, continuous_action_space = True):

        # Action Space
        self.max_price = 5
        self.step_size = 1
        self.continuous_action_space = continuous_action_space
        
        self.action_space = Box(low = 0, high = self.max_price, shape = (1,)) if self.continuous_action_space else Discrete(int(self.max_price / self.step_size) + 1)
        self.action_space_max = int(self.max_price / self.step_size)

        # Observation Space
        self.booking_time = 5
        self.flight_capacity = 5
        # we cannot take our own price as a value of the state space, right?
        self.observation_space = MultiDiscrete([self.booking_time + 1, self.flight_capacity + 1, self.flight_capacity + 1, self.action_space_max + 1])


        # Event Space
        self.customers = Customers(['family'], self.max_price, self.booking_time)
        self.customers_per_round = 5
        self.event_space = MultiDiscrete([self.customers_per_round + 1, self.customers_per_round + 1, self.customers_per_round + 1])

        self.stochastic_customers = True
        self.edgeworth = False

        self.initial_state = [0, self.flight_capacity, self.flight_capacity,  int(self.action_space_max / 2)]

        self.reset()

    def random_action(self):
        return random.random() * self.max_price if self.continuous_action_space else random.randrange(0, int(self.max_price / self.step_size))


    def transform_action(self, a):
        return a * self.step_size if not self.continuous_action_space else a


    def calculate_p_cust(self, a, timestep):
        _a = self.transform_action(a) if np.isscalar(a) else a
        return self.customers.calculate_p(_a, timestep)
    
    def calculate_p(self, a, s):
        own_p = self.calculate_p_cust(a, s[0])
        comp_p = self.calculate_p_cust(s[3], s[0])
        if self.edgeworth:
            p = [1, 0, 0] if own_p > comp_p else [0, 1, 0]
        else:
            p = softmax([own_p, comp_p, max(1 - own_p - comp_p, 0)])
        return p


    def get_event_p(self, i, a, s):
        p = self.calculate_p(a, s)
        return multinomial.pmf(i, self.customers_per_round, p)


    def sample_event(self, a , s):
        p = self.calculate_p(a, s)
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, p)
        else:
            i = []
            i.append(int(self.customers_per_round * p[0]))
            i.append(int(self.customers_per_round * p[1]))
            i.append(self.customers_per_round - i[0] - i[1])
            return i
    
    
    def get_reward(self, i, a, s):
        return self.transform_action(a) * min(i[0], s[1])
    

    def transit_state(self, i, a, s):
        comp_price = a - 1 if a > 2 else 4
        return [s[0] + 1, max(0, s[1] - i[0]), max(0, s[2] - i[1]), comp_price]
    

    def step(self, a):
        a = int(a[0]) if isinstance(a, np.ndarray) else int(a)
        i = self.sample_event(a, self.s)
        reward = self.get_reward(i, a, self.s)
        comp_rew = i[1] * self.s[3]
        self.s = self.transit_state(i, a, self.s)

        return self.s, reward, self.s[0] == self.booking_time - 1, {'comp_rew': comp_rew, 'i': i}


    def reset(self):
        self.s = self.initial_state
        return self.initial_state