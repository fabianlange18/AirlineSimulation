# TODO: Adapt for using multiple dimensions

import numpy as np

from .solver import Solver


class PolicyIteration(Solver):
    def solve(self):
        """
        This should iteratively perform value and policy improvements until convergence
        is achieved.
        """
        done = False
        counter = 0
        while not done:
            self.impr_values()
            done = self.impr_policy()
            counter += 1
        print(f"Convergence after {counter} iterations.")

    def impr_policy(self):
        """
        This should update the policy table self.policy given the current value estimations.
        It should return True if and only if an action changed (this can be
        used for dynamically determining if the policy is stable).
        """
        previous = self.policy.copy()
        for s in self.env.observation_space:
            values = []
            for a in self.env.action_space:
                value = 0
                for i in self.env.event_space:
                    value += self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * self.value[self.env.transit_state(i, a, s)])
                values.append(value)
            self.policy[s] = values.index(max(values))
        return np.all(previous == self.policy)

    def impr_values(self):
        """
        This should update the internal value table self.value based on the current policy.
        """
        t = 1000
        while t > 0:
            t -= 1
            for s in self.env.observation_space:
                a = self.policy[s]
                value = 0
                for i in self.env.event_space:
                    value += self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * self.value[self.env.transit_state(i, a, s)])
                self.value[s] = value