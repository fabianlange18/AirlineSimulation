import random
import numpy as np
from solver import Solver

class ADP(Solver):
    def solve(self, K):
        s = self.env.initial_state
        for k in range(K):
            if random.random() < self.eps:
                a = self.env.action_space.sample()
            else:
                values = []
                for a in self.env.action_space:
                    values.append(
                        sum(
                            self.env.get_event_p(i, a, s) + (self.env.get_reward(i, a, s) + self.gamma * self.value[self.env.transit_state(i, a, s)]) for i in self.env.event_space
                        )
                    )
                self.policy[s] = values.index(max(values))
                a = self.policy[s]
            value_candidate = sum(
                self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * self.value[self.env.transit_state(i, a, s)])
                for i in self.env.event_space
            )
            self.value[s] = max(self.value[s], value_candidate)
            s = self.env.transit_state(self.env.event_space.sample(), a, s)
