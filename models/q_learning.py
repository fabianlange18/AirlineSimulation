# TODO: Adapt for using multiple dimensions

from tqdm import tqdm
import random
import numpy as np

from solver import Solver

class Q_Learning(Solver):

    def __init__(self, env, debug=True):
        self.q_values = None
        super().__init__(env, debug)

    def solve(self):
        self.env.reset()
        K = self.env.time_limit * 1000
        states_trajectory = []
        rewards_trajectory = []
        for k in tqdm(range(K)):
            s = self.env.s
            states_trajectory.append(s)
            a = self.choose_action(s)
            s_next, reward, done, _ = self.env.step(a)
            rewards_trajectory.append(reward)
            learning_rate = self.learning_rate_schedule(k, K)
            self.q_values[s][a] = learning_rate * (reward + self.gamma * max(self.q_values[s_next]) - self.q_values[s][a]) + self.q_values[s][a]
        self.value = np.max(self.q_values, axis = 1)
        self.policy = np.argmax(self.q_values, axis = 1)
        return states_trajectory, rewards_trajectory
    
    def choose_action(self, s):
        if random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[s])
        
    def learning_rate_schedule(self, k, K):
        return 1 - (k / K)

    def reset(self):
        self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        super().reset()
