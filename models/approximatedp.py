import random
from tqdm import trange

from .solver import Solver

class ADP(Solver):
    def solve(self, steps):
        s = self.env.s
        a_comp = self.env.rule_based_competitor()
        print("\nTabular Forward Calculation")
        for _ in trange(steps):
            if random.random() < self.eps:
                a = self.env.action_space.sample()
                i_comp = self.env.sample_event([a, a_comp], s)[1]
                value_candidate = sum(
                    self.env.get_event_p([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s) * (self.env.get_reward([i, i_comp], [a, a_comp], s)[0] + self.gamma *
                                                     ( self.value[*self.env.transit_state([i, i_comp], [self.policy[*s], a_comp], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                     )
                    for i in range(self.env.customers_per_round)
                )
            else:
                # ok with multiple calls of sample_event to get i_comp?
                a_max = max((
                    (a, sum(
                        self.env.get_event_p([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s) * (self.env.get_reward([i, self.env.sample_event([a, a_comp], s)[1]], [a, a_comp], s)[0] + self.gamma *
                                                         ( self.value[*self.env.transit_state([i, self.env.sample_event([a, a_comp], s)[1]], self.policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                         )
                        for i in range(self.env.customers_per_round)
                    ))
                    for a in range(self.env.action_space_max)
                ), key=lambda o: o[1])
                a = a_max[0]
                value_candidate = a_max[1]
                # TODO: Question - Do we need to update the policy also when we play a random action?
                self.policy[*s] = a
            
            self.value[*s] = max(self.value[*s], value_candidate)
            
            if s[0] == self.env.booking_time - 1:
                s = self.env.reset()
            else:
                s = self.env.transit_state(self.env.sample_event(a, s), a, s)
