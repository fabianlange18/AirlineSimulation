import gym
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

import wandb

import numpy as np
from scipy.stats import multinomial

class AirlineEnvironment(gym.Env):

    def __init__(self, continuous_action_space = False) -> None:

        # Observation Space
        self.booking_time = 50
        self.flight_capacity = 100
        self.observation_space = MultiDiscrete([self.booking_time, self.flight_capacity])

        # Action Space
        self.max_price = 250
        self.step_size = 10 # only relevant in discrete case
        self.continuous_action_space = continuous_action_space
        self.action_space = Box(low = 0, high = self.max_price, shape = (1,)) if self.continuous_action_space else Discrete(self.max_price / self.step_size)

        # Event Space
        self.customers_per_round = 100
        self.event_space = Discrete(self.customers_per_round)

        self.stochastic_customers = False

        self.reset()


    def transform_action(self, a):
        return a * self.step_size
    
    def get_p(self, a, timestep):
        return (1 - self.transform_action(a) / self.max_price) * (1 + timestep) / self.booking_time

    def get_event_p(self, i, a, s):
        p = self.get_p(a, s[0])
        probability_dist = [p, 1-p]
        # Do we need to change this for stochastic customers?
        return multinomial.pmf([i, self.customers_per_round - 1], self.customers_per_round, probability_dist)

    def sample_event(self, a, s):
        p = self.get_p(a, s[0])
        probability_dist = [p, 1-p]
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, probability_dist)[0]
        else:
            return int(np.multiply(probability_dist, self.customers_per_round)[0])
    
    def get_reward(self, i, a, s):
        return self.transform_action(a) * min(i, s[1])
    
    def transit_state(self, i, a, s):
        return [s[0] + 1, max(0, s[1] - i)]

    def step(self, a):
        a = a[0]
        i = self.sample_event(a, self.s)
        self.s = self.transit_state(i, a, self.s)
        reward = self.get_reward(i, a, self.s)

        wandb.log({
            'price' : a,
            'empty_seats' : self.s[1],
            'buying_customers' : i,
            'profit' : reward
        })
        return self.s, reward, self.s[0] == self.booking_time , {}

    def simulate_policy(self, policy):
        self.reset()
        actions = []
        tickets_available = []
        rewards = []
        while self.s[0] < self.booking_time:
            actions.append(self.transform_action(policy[self.s[0]][self.s[1]]))
            tickets_available.append(self.s[1])
            _, reward, _, _ = self.step(actions[-1])
            rewards.append(reward)
        return actions, tickets_available, rewards

    def reset(self):
        self.s = [0, self.flight_capacity - 1]
        return self.s