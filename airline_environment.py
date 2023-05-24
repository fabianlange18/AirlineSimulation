import numpy as np

import gymnasium as gym
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
            [[0, 1000, 150, 600, 130, 120],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]
        )

        self.observation_space = gym.spaces.Discrete(self.n_flights)
        self.action_space = gym.spaces.MultiDiscrete([len(self.airports), self.max_price])

        self.state = 1

    def step(self, action: Any):
        demand = self.demand[0][action[0]]
        reward = 0
        if self.customer_behavior(action[1]):
            self.demand[0][action[0]] = min(0, demand - self.flight_capacity)
            reward += min(demand, self.flight_capacity) * action[1]
        else:
            print("Price too high")
        return self.state, reward, False, {}
    

    def customer_behavior(price):
        return price < 500