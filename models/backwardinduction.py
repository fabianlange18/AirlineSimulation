import numpy as np
from .solver import Solver


class BackwardInduction(Solver):

    def comp_expected_reward(self, t, future):
        r = np.zeros((self.env.flight_capacity,))
        a = np.zeros((self.env.flight_capacity,))
        for s in range(self.env.flight_capacity):
            s = np.array([t, s])
            a_max = max((
                (a, sum(
                    self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + future[self.env.transit_state(i, a, s)[1]])
                    for i in range(self.env.customers_per_round)
                ))
                for a in range(self.env.action_space_max)
            ), key=lambda o: o[1])
            r[s[1]] = a_max[1]
            a[s[1]] = a_max[0]
        return r, a

    def solve(self):
        self.reset()
        for t in range(self.env.booking_time - 1, -1, -1):
            self.value[t], self.policy[t] = self.comp_expected_reward(t, self.value[t+1] if t + 1 < self.env.booking_time else np.zeros((self.env.flight_capacity,)))