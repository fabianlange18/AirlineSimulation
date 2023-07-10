
from models.solver import Solver


class InitialValueCalculator(Solver):

    def calculate_initial_value(self, policy):
        delta = self.max_delta
        while delta >= self.max_delta:
            delta = 0
            for s in self.possible_states_array:
                a_comp = self.env.rule_based_competitor()
                v = self.value[*s]
                self.value[*s] = sum((
                    self.env.get_event_p([i, self.env.sample_event([policy[*s], a_comp], s)[1]], [policy[*s], a_comp], s) * (self.env.get_reward([i, self.env.sample_event([policy[*s], a_comp], s)[1]], [policy[*s], a_comp], s)[0] + self.gamma *
                                                                   ( self.value[*self.env.transit_state([i, self.env.sample_event([policy[*s], a_comp], s)[1]], [policy[*s], a_comp], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                                   )
                    for i in range(self.env.customers_per_round)
                ))
                delta = max(delta, abs(v - self.value[*s]))
        
        return self.value[*self.env.initial_state]