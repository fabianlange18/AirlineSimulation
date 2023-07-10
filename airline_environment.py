import gym
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

# import wandb

import numpy as np
from scipy.stats import multinomial
from scipy.special import softmax


class AirlineEnvironment(gym.Env):

    def __init__(self, continuous_action_space=True) -> None:

        # Observation Space
        self.booking_time = 10
        self.flight_capacity = 12
        self.observation_space = MultiDiscrete([self.booking_time, self.flight_capacity])

        # Action Space
        self.max_price = 20
        self.step_size = 1  # only relevant in discrete case
        self.continuous_action_space = continuous_action_space
        self.action_space = Box(low=0, high=self.max_price, shape=(1,)) if self.continuous_action_space else Discrete(
            int(self.max_price / self.step_size))
        self.action_space_max = int(self.max_price / self.step_size)

        # Event Space
        self.customers_per_round = 10
        self.event_space = Discrete(self.customers_per_round)

        self.stochastic_customers = False

        self.initial_state = [0, self.flight_capacity - 1]

        self.reset()

    def transform_action(self, a):
        return a * self.step_size

    def get_p_dist(self, a, timestep):
        p = (1 - self.transform_action(a) / self.max_price) * (1 + timestep) / self.booking_time
        return [p, 1 - p]

    def get_event_p(self, i, a, s):
        p_dist = self.get_p_dist(a, s[0])
        # Do we need to change this for stochastic customers?
        return multinomial.pmf([i, self.customers_per_round - i], self.customers_per_round, p_dist)

    def sample_event(self, a, s):
        p_dist = self.get_p_dist(a, s[0])
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, p_dist)[0]
        else:
            return int(np.multiply(p_dist, self.customers_per_round)[0])

    def get_reward(self, i, a, s):
        return self.transform_action(a) * min(i, s[1])

    def transit_state(self, i, a, s):
        return [s[0] + 1, max(0, s[1] - i)]

    def step(self, a):
        a = a[0] if isinstance(a, np.ndarray) else a
        i = self.sample_event(a, self.s)
        reward = self.get_reward(i, a, self.s)
        self.s = self.transit_state(i, a, self.s)

        # wandb.log({
        #     'price' : a,
        #     'empty_seats' : self.s[1],
        #     'buying_customers' : min(i, self.s[1]),
        #     'profit' : reward
        # })

        return self.s, reward, self.s[0] == self.booking_time - 1, {}

    def reset(self):
        self.s = self.initial_state
        return self.initial_state


class AirlineDuopoly(AirlineEnvironment):
    def __init__(self, continuous_action_space=True) -> None:
        super().__init__(continuous_action_space)
        self.initial_state = self.initial_state = [0, self.flight_capacity - 1, self.flight_capacity - 1]
        self.observation_space = MultiDiscrete(
            [self.booking_time, self.flight_capacity, self.flight_capacity])

    def get_p_dist(self, a, timestep):
        p_1 = (1 - self.transform_action(a[0]) / self.max_price) * (1 + timestep) / self.booking_time
        p_2 = (1 - self.transform_action(a[1]) / self.max_price) * (1 + timestep) / self.booking_time
        p_3 = 1 - p_1 - p_2
        p = softmax([p_1, p_2, p_3])
        return p

    def get_event_p(self, i, a, s):
        p_dist = self.get_p_dist(a, s[0])
        return multinomial.pmf([i[0], i[1], self.customers_per_round - i[0] - i[1]], self.customers_per_round, p_dist)

    def sample_event(self, a, s):
        p_dist = self.get_p_dist(a, s[0])
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, p_dist)[0:2]
        else:
            return np.asarray(np.multiply(p_dist, self.customers_per_round)[0:2], dtype=int)

    def get_reward(self, i, a, s):
        reward_1 = super().get_reward(i[0], a[0], s)
        reward_2 = super().get_reward(i[1], a[1], s)
        return [reward_1, reward_2]

    def transit_state(self, i, a, s):
        return [s[0] + 1, max(0, s[1] - i[0]), max(0, s[2] - i[1])]

    def step(self, a_player):
        a = [a_player[0] if isinstance(a_player, np.ndarray) else a_player, self.rule_based_competitor()]
        i = self.sample_event(a, self.s)
        rewards = self.get_reward(i, a, self.s)
        self.s = self.transit_state(i, a, self.s)
        output_dict = {}
        output_dict["comp/reward_per_step"] = rewards[1]

        return self.s, rewards[0], self.s[0] == self.booking_time - 1, output_dict

    def rule_based_competitor(self):
        if self.s[0] < 6:
            a = 16
        elif self.s[0] > 6 and self.s[0] > 8:
            a = 15
        else:
            a = 13

        return a
