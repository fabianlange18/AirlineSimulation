import numpy as np
from itertools import product

def setup_possible_states_array(env):
        ranges = [range(max_val) for max_val in env.observation_space.nvec]
        return np.array(list(product(*ranges)))

def setup_possible_events_array(env):
        ranges = [range(max_val) for max_val in env.event_space.nvec]
        possible_events = np.array(list(product(*ranges)))
    
        # Filter out values where the sum is not equal to 5
        filtered_events = possible_events[np.sum(possible_events, axis=1) == env.customers_per_round]
    
        return filtered_events