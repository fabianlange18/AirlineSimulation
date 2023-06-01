# TODO: Adapt for using multiple dimensions

import numpy as np

from .solver import Solver


class ValueIteration(Solver):
    def solve(self, print_diff_to=None):
        self.reset()
        delta = self.eps
        i = 0
        while delta >= self.eps:
            i += 1
            delta = 0
            for s in self.env.observation_space:
                v = self.value[s]
                action_values = (
                    (a,
                    sum((
                        self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * self.value[
                            self.env.transit_state(i, a, s)])
                        for i in self.env.event_space
                    )))
                    for a in self.env.action_space
                )
                max_a = max(action_values, key=lambda o: o[1])
                self.value[s] = max_a[1]
                self.policy[s] = max_a[0]
                delta = max(delta, abs(v - self.value[s]))

            if print_diff_to is not None:
                diff = np.max(np.abs(self.value - print_diff_to))
                print(f"DIFF: {diff} DIFF-D: {np.abs(diff-delta)} DIFF/D: {diff/delta}")
            not self.debug or print(f"I {i} D {delta}")
