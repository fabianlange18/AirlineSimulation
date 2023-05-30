import gym
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box

import numpy as np

class AirlineEnvironment(gym.Env):


    def __init__(self) -> None:

        # Initial State
        # self.timestep = 0
        # self.booked_seats = 0

        # Observation Space
        self.booking_time = 50
        self.flight_capacity = 500
        self.observation_space = MultiDiscrete([self.booking_time, self.flight_capacity])

        # Action Space
        self.max_price = 2000
        self.action_space = Box(low = 0, high = self.max_price, shape = (1,))

        self.customers_per_round = 10


    def step(self, action):

        # time_until_takeoff = self.booking_time - self.timestep % self.booking_time
        price = action[0]
        empty_seats = self.flight_capacity - self.state[1] - 1

        buying_customers = self.simulate_customers(price, self.state[0])

        buying_customers = min(empty_seats, buying_customers)

        reward = buying_customers * price

        self.state[0] += 1
        self.state[1] += buying_customers

        return self.state, reward, self.state[0] == self.booking_time , {}


    def simulate_customers(self, price, timestep):
        p = (1 - price / self.max_price) * (1 + timestep) / self.booking_time
        probability_distribution = [p, 1-p]
        buying_customers, _ = np.random.multinomial(self.customers_per_round, probability_distribution)
        return buying_customers


    def reset(self):
        self.state = [0, 0]
        return self.state