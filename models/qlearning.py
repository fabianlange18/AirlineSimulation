import numpy as np
from tqdm import trange

from .solver import Solver


class QLearning(Solver):
    def __init__(self, env, debug = True):
        super().__init__(env, debug)
        self.lr = 0.8
        self.eps = 0.5
        # Use this variable to store the bellman error in every update.
        self.bellman_errors = []
        self.q_value = np.zeros((*self.env.observation_space.nvec, self.env.action_space_max))

    def solve(self, steps):
        s = self.env.reset()
        print("\nQ-Learning")
        for _ in trange(steps):
            a = self.env.action_space.sample() if np.random.random() < self.eps else np.argmax(self.q_value[*s])
            s_old = s
            s, r, done, _ = self.env.step(a)

            bellman_err = r + (0 if done else (self.gamma * np.max(self.q_value[*s]))) - self.q_value[*s_old, a]
            self.bellman_errors.append(bellman_err)
            self.q_value[*s_old, a] = self.lr * bellman_err + self.q_value[*s_old, a]
            
            self.lr *= 0.99999
            self.lr = max(0.01, self.lr)
            self.eps *= 0.9999
            self.eps = max(0.1, self.eps)

            if done:
                s = self.env.reset()

        self.policy = np.argmax(self.q_value, axis=-1)

    def render_solution(self):
        print("# S # V # A #")
        for k in self.env.observation_space:
            print(f"# {k} # {np.max(self.q_value[k])} # {self.policy[k]}")
