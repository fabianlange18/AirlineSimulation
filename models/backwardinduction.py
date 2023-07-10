import numpy as np
from .solver import Solver


class BackwardInduction(Solver):

    def comp_expected_reward(self, t, future):
        r = np.zeros((self.env.flight_capacity,))
        a = np.zeros((self.env.flight_capacity,))
        a_comp = self.env.rule_based_competitor()
        for s in range(self.env.flight_capacity):
            s = np.array([t, s, s])
            # a is not array here?
            #i_comp = self.env.sample_event([a, a_comp], s)[1]
            """a_max = max((
                (action, sum(
                    self.env.get_event_p([i, self.env.sample_event([action, a_comp], s)[1]], [action, a_comp], s) * (self.env.get_reward([i, self.env.sample_event([action, a_comp], s)[1]], [action, a_comp], s)[0] + future[self.env.transit_state([i, self.env.sample_event([action, a_comp], s)[1]], [action, a_comp], s)[1]])
                    for i in range(self.env.customers_per_round)
                ))
                for action in range(self.env.action_space_max)
            ), key=lambda o: o[1])"""
            a_max = max((
                (a, sum(
                    self.env.get_event_p([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s) * (
                                self.env.get_reward([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s)[0] +
                                future[
                                    self.env.transit_state([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp],
                                                           s)[1]])
                    for i in range(self.env.customers_per_round)
                ))
                for a in range(self.env.action_space_max)
            ), key=lambda o: o[1])
            print(a_max)
            r[s[1]] = a_max[1]
            a[s[1]] = a_max[0]
        return r, a

    def solve(self):
        self.reset()
        for t in range(self.env.booking_time - 1, -1, -1):
            self.value[t], self.policy[t] = self.comp_expected_reward(t, self.value[t+1] if t + 1 < self.env.booking_time else np.zeros((self.env.flight_capacity,)))
