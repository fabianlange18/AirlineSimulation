from .solver import Solver


class PolicyIteration(Solver):
    def solve(self):
        self.reset()
        instable = True
        i = 0
        while instable:
            not self.debug or print(f"Solving in iter #{i}")
            i += 1
            self.impr_values()
            not self.debug or print(f"Value improvement done")
            instable = self.impr_policy()
        return self.policy

    def impr_policy(self):
        p_instable = False
        for s in self.possible_states_array:
            a = self.policy[*s]
            action_values = [
                (a,
                sum((
                    self.env.get_event_p(i, a, s) *
                    (self.env.get_reward(i, a, s) + self.gamma * 
                     ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                     )
                    for i in range(self.env.customers_per_round)
                )))
                for a in range(self.env.action_space_max)
            ]
            self.policy[*s] = max(action_values, key=lambda o: o[1])[0]
            p_instable = p_instable or (a != self.policy[*s])

        return p_instable

    def impr_values(self):
        delta = self.max_delta
        while delta >= self.max_delta:
            delta = 0
            for s in self.possible_states_array:
                v = self.value[*s]
                self.value[*s] = sum((
                    self.env.get_event_p(i, self.policy[*s], s) * (self.env.get_reward(i, self.policy[*s], s) + self.gamma * 
                                                                   ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                                   )
                    for i in range(self.env.customers_per_round)
                ))
                delta = max(delta, abs(v - self.value[*s]))
