import random
from .solver import Solver

class ADP(Solver):
    def solve(self, K):
        s = self.env.s
        for _ in range(K):
            if random.random() < self.eps:
                a = self.env.action_space.sample()
                value_candidate = sum(
                    self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * 
                                                     ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                     )
                    for i in range(self.env.customers_per_round)
                )
            else:
                a_max = max((
                    (a, sum(
                        self.env.get_event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * 
                                                         ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                         )
                        for i in range(self.env.customers_per_round)
                    ))
                    for a in range(self.env.action_space_max)
                ), key=lambda o: o[1])
                a = a_max[0]
                value_candidate = a_max[1]
                # TODO: Question - Do we need to update the policy also when we play a random action?
                self.policy[s[0]][s[1]] = a
            
            self.value[s[0]][s[1]] = max(self.value[s[0]][s[1]], value_candidate)
            
            if s[0] == self.env.booking_time - 1:
                s = self.env.reset()
            else:
                s = self.env.transit_state(self.env.sample_event(a, s), a, s)
