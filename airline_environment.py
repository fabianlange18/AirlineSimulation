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
        self.booking_time = 11
        self.flight_capacity = 6
        self.observation_space = MultiDiscrete([self.booking_time, self.flight_capacity])

        # Action Space
        self.max_price = 60
        self.step_size = 10  # only relevant in discrete case
        self.continuous_action_space = continuous_action_space
        self.action_space = Box(low=0, high=self.max_price, shape=(1,)) if self.continuous_action_space else Discrete(
            int(self.max_price / self.step_size))
        self.action_space_max = int(self.max_price / self.step_size)

        # Event Space
        self.customers_per_round = 2
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
        self.initial_state = self.initial_state = [0, self.flight_capacity - 1, self.flight_capacity - 1, 0, 0]
        self.observation_space = MultiDiscrete(
            [self.booking_time, self.flight_capacity, self.flight_capacity, self.max_price, self.max_price])

    def get_p_dist(self, a, timestep):
        p_1 = a[1] / (a[0] + a[1])
        p_2 = a[0] / (a[0] + a[1])
        p_3 = 2 - p_1 - p_2
        p = softmax([p_1, p_2, p_3])
        #print("action ", a)
        #print("p dist ", p)
        return p

    def get_event_p_player(self, i_player, a, s):
        p = self.get_p_dist(a, s[0])[0]
        if i_player == 0:
            return 1 - p
        else:
            return p

    def get_event_p_comp(self, i_comp, a, s):
        p = self.get_p_dist(a, s[0])[1]
        if i_comp == 0:
            return 1 - p
        else:
            return p
    
    def sample_event(self, a, s):
        p_player = int(self.get_event_p_player(1, a, s))
        p_comp = int(self.get_event_p_player(1, a, s))

        # similar offers
        if p_player == p_comp == 1:
            if np.random.random() > 0.5:
                return [0, 1]
            else:
                return [1, 0]
        # player sells
        if p_player == 1 and p_comp == 0:
            return [1, 0]
        # comp sells
        if p_player == 0 and p_comp == 1:
            return [0, 1]
        # no one sells
        if p_player == p_comp == 0:
            return [0, 0]

    """

    def get_event_p(self, i, a, s):
        p_dist = self.get_p_dist(a, s[0])
        #print("i_p ", i)
        #print("event p ", multinomial.pmf([i[0], i[1], self.customers_per_round - i[0] - i[1]], self.customers_per_round, p_dist))
        return multinomial.pmf([i[0], i[1], self.customers_per_round - i[0] - i[1]], self.customers_per_round, p_dist)


    def sample_event(self, a, s):
        p_dist = self.get_p_dist(a, s[0])
        if self.stochastic_customers:
            return np.random.multinomial(self.customers_per_round, p_dist)[0:2]
        else:
            #print("sampled event ", np.asarray(np.multiply(p_dist, self.customers_per_round)[0:2], dtype=int))
            return np.asarray(np.multiply(p_dist, self.customers_per_round)[0:2], dtype=int)

"""

    def get_reward(self, i, a, s):
        reward_1 = a[0] * min(i[0], s[1])
        reward_2 = a[1] * min(i[1], s[2])
        #print("rew events", i)
        #print("rew actions ", a)
        #print("rewards ", [reward_1, reward_2])
        return [reward_1, reward_2]

    def transit_state(self, i, a, s):
        #print("state last state ",s)
        #print("state events ", i)
        #print("state actions ", a)
        #print("next state is ", [s[0] + 1, max(0, s[1] - i[0]), max(0, s[2] - i[1]), a[0], a[1]])
        return [s[0] + 1, max(0, s[1] - i[0]), max(0, s[2] - i[1]), a[0], a[1]]

    def step(self, a_player):
        a = [a_player[0] if isinstance(a_player, np.ndarray) else a_player, self.rule_based_competitor(self.s)]
        i = self.sample_event(a, self.s)
        rewards = self.get_reward(i, a, self.s)
        self.s = self.transit_state(i, a, self.s)
        output_dict = {"comp/reward_per_step": rewards[1]}

        return self.s, rewards[0], self.s[0] == self.booking_time - 1, output_dict

    def rule_based_competitor(self, s):
        if s[2] > 0:
            a = 20
        else:
            a = 50
        return a
