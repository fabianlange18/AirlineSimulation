import random
from tqdm import trange

from .solver import Solver

class ADP(Solver):
    def solve(self, steps, prog_bar = True):
        self.reset()
        
        s = self.env.s
        if prog_bar:
            print("\nTabular Forward Calculation")
        iteration_range = trange(steps) if prog_bar else range(steps)

        for _ in iteration_range:
            if random.random() < self.eps:
                a = self.env.action_space.sample()
                value_candidate = sum(
                    self.event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * 
                                                     ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                     )
                    for i in range(self.env.customers_per_round)
                )
            else:
                a_max = max((
                    (a, sum(
                        self.event_p(i, a, s) * (self.env.get_reward(i, a, s) + self.gamma * 
                                                         ( self.value[*self.env.transit_state(i, self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                         )
                        for i in range(self.env.customers_per_round)
                    ))
                    for a in range(self.env.action_space_max)
                ), key=lambda o: o[1])
                a = a_max[0]
                value_candidate = a_max[1]
                self.policy[*s] = a
            
            self.value[*s] = max(self.value[*s], value_candidate)
            
            if s[0] == self.env.booking_time - 1:
                s = self.env.reset()
            else:
                s = self.env.transit_state(self.env.sample_event(a, s), a, s)
