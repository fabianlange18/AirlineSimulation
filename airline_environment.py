import gym

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

import random
import numpy as np
from scipy.stats import multinomial

from customers import Customers

class AirlineEnvironment(gym.Env):

    def __init__(self, continuous_action_space = True):

        # Observation Space
        self.booking_time = 100
        self.flight_capacity = 100
        self.observation_space = MultiDiscrete([self.booking_time + 1, self.flight_capacity + 1])

        # Action Space
        self.max_price = 20
        self.step_size = 1
        self.continuous_action_space = continuous_action_space

        self.action_space = Box(low = 0, high = self.max_price, shape = (1,)) if self.continuous_action_space else Discrete(int(self.max_price / self.step_size) + 1)
        self.action_space_max = int(self.max_price / self.step_size)

        # Event Space
        self.customers = Customers(['rational', 'family', 'business', 'party', 'early_booking'], self.max_price, self.booking_time)
        self.customers_per_round = 5
        self.event_space = Discrete(self.customers_per_round + 1)

        self.stochastic_customers = True

        self.initial_state = [0, self.flight_capacity]

        self.reset()

    def random_action(self):
        return random.random() * self.max_price if self.continuous_action_space else random.randrange(0, int(self.max_price / self.step_size))


    def transform_action(self, a):
        return a * self.step_size if not self.continuous_action_space else a


    def calculate_p(self, a, timestep):
        _a = self.transform_action(a) if np.isscalar(a) else a
        return self.customers.calculate_p(_a, timestep)


    def get_event_p(self, i, a, s):
        p = self.calculate_p(a, s[0])
        return multinomial.pmf([i, self.customers_per_round - i], self.customers_per_round, [p, 1-p])


    def sample_event(self, a , s):
        p = self.calculate_p(a, s[0])
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, [p, 1-p])[0]
        else:
            return int(self.customers_per_round * p)
    
    
    def get_reward(self, i, a, s):
        return self.transform_action(a) * min(i, s[1])
    

    def transit_state(self, i, a, s):
        return [s[0] + 1, max(0, s[1] - i)]
    

    def step(self, a):
        a = a[0] if isinstance(a, np.ndarray) else a
        i = self.sample_event(a, self.s)
        reward = self.get_reward(i, a, self.s)
        self.s = self.transit_state(i, a, self.s)

        return self.s, reward, self.s[0] == self.booking_time - 1, {'i': i}


    def reset(self):
        self.s = self.initial_state
        return self.initial_state