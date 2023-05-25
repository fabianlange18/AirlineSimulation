import numpy as np

import gym
from typing import Any, SupportsFloat

class AirlineEnvironment(gym.Env):


    def __init__(self) -> None:

        self.airport_names = ['BER', 'JFK', 'GIG', 'HND', 'CPT', 'SYD']
        self.airports = np.arange(len(self.airport_names))

        self.airport_distances = np.array(
            [[0, 5000, 8000, 15000, 6000, 20000],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]
        )

        self.var_flight_costs = 1
        # Fix cost do not change this much: self.fix_flight_costs = 1000

        self.max_price = 1000
        self.flight_capacity = 200

        self.n_flights = 1
        
        self.demand = np.array(
            [[0, 1000000, 1000000, 1000000, 1000000, 1000000],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]
        )

        self.state = 0
        self.max_state = 6

        self.observation_space = gym.spaces.discrete.Discrete(self.max_state)
        self.action_space = gym.spaces.box.Box(low=0.0, high=self.max_price, shape=(1,))


    def step(self, action: Any):
        demand = self.demand[0][self.state]
        reward = 0
        if self.customer_behavior(action[0]):
            reward += min(demand, self.flight_capacity) * action[0]
            # self.demand[0][self.state] = max(0, demand - self.flight_capacity)
        # else:
            # print("Price too high")
        self.state += 1
        return self.state, reward, self.state == self.max_state , {}


    def customer_behavior(self, price):
        return price < 500


    def reset(self):
        self.state = 0
        return self.state