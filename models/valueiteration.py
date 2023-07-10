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
                v = self.value[*s]
                action_values = (
                    (a,
                    sum((
                        self.event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * 
                                                         ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                         )
                        for i in range(self.env.customers_per_round + 1)
                    )))
                    for a in range(self.env.action_space_max + 1)
                )
                max_a = max(action_values, key=lambda o: o[1])
                self.value[*s] = max_a[1]
                self.policy[*s] = max_a[0]
                delta = max(delta, abs(v - self.value[*s]))

            not self.debug or print(f"I {i} D {delta}")
