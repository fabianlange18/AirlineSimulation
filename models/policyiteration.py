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

    def impr_policy(self):
        p_instable = False
        for s in self.possible_states_array:
            a = self.policy[*s]
            a_comp = self.env.rule_based_competitor(s)
            action_values = [
                (a,
                 sum((
                     self.env.get_event_p([i, 0], [a, 19], s) *
                     (self.env.get_reward([i, 0], [a, 19], s)[0] + self.gamma *
                      (self.value[*self.env.transit_state([i, 0], [int(self.policy[*s]), 19], s)] if s[
                                                                                         0] < self.env.booking_time - 1 else 0)
                      )
                     for i in range(self.env.customers_per_round)
                 )))
                for a in range(self.env.action_space_max)
            ]
            print("action is", action_values)
            self.policy[*s] = max(action_values, key=lambda o: o[1])[0]
            p_instable = p_instable or (a != self.policy[*s])

        return p_instable

    def impr_values(self):
        delta = self.max_delta
        while delta >= self.max_delta:
            delta = 0
            for s in self.possible_states_array:
                v = self.value[*s]
                a_comp = self.env.rule_based_competitor(s)
                self.value[*s] = sum((
                    self.env.get_event_p([i, 0], [self.policy[*s], 19], s) * (
                                self.env.get_reward([i, 0], [self.policy[*s], 19], s)[0] + self.gamma *
                                (self.value[*self.env.transit_state([i, 0], [int(self.policy[*s]), 19], s)] if s[
                                                                                                   0] < self.env.booking_time - 1 else 0)
                                )
                    for i in range(self.env.customers_per_round)
                ))
                print("value is ", self.value[*s])
                delta = max(delta, abs(v - self.value[*s]))
                #print(delta)
