
from models.solver import Solver


class InitialValueCalculator(Solver):

    def calculate_initial_value(self, policy):

        self.reset()

        delta = self.max_delta
        while delta >= self.max_delta:
            delta = 0
            for s in self.possible_states_array:
                v = self.value[*s]
                self.value[*s] = sum((
                    self.event_p(i, policy[*s], s) * (self.env.get_reward(i, policy[*s], s) + self.gamma * 
                                                                   ( self.value[*self.env.transit_state(i, policy[*s], s)] if s[0] < self.env.booking_time - 1 else 0)
                                                                   )
                    for i in self.possible_events_array
                ))
                delta = max(delta, abs(v - self.value[*s]))
        
        return self.value[*self.env.initial_state]