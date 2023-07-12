import numpy as np

from .solver import Solver


class ValueIteration(Solver):
    def solve(self):
        self.reset()
        delta = self.eps
        i = 0
        while delta >= self.eps:
            i += 1
            delta = 0
            for s in self.possible_states_array:
                #print(self.value.shape)
                #print(*self.env.transit_state([0, self.env.sample_event([2, 3], s)[1]],
                                              #[self.policy[*s], 3], s))
                #print(self.env.transit_state([0, self.env.sample_event([2, 3], s)[1]],
                                              #[self.policy[*s], 3], s))
                #print([0, self.env.sample_event([2, 3], s)][1])
                #print([self.policy[*s], 3])
                #print(s)
                v = self.value[*s]
                a_comp = self.env.rule_based_competitor(s)
                action_values = (
                    (a,
                     sum((
                         self.env.get_event_p([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s) * (
                                 self.env.get_reward([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s)[
                                     0] + self.gamma *
                                 (self.value[*self.env.transit_state([i, self.env.sample_event([a, a_comp], s)[1]],
                                                                     [int(self.policy[*s]), a_comp], s)] if s[
                                                                                                           0] < self.env.booking_time - 1 else 0)
                         )
                         for i in range(self.env.customers_per_round)
                     )))
                    for a in range(self.env.action_space_max)
                )
                max_a = max(action_values, key=lambda o: o[1])
                self.value[*s] = max_a[1]
                self.policy[*s] = max_a[0]
                delta = max(delta, abs(v - self.value[*s]))
                print("max_a ", max_a)
                print("delta ", delta)

            not self.debug or print(f"I {i} D {delta}")
