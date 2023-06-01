# TODO: Adapt for using multiple dimensions

import numpy as np


class BackwardInduction(object):
    def __init__(self, env):
        self.env    = env
        self.value  = None #timesteps  * #state = values  (result of value function)
        self.policy = None #timesteps  * #state = actions (take the best possible at this state)

    def reset(self):
        self.value = np.zeros((self.env.time_limit+1, self.env.observation_space.n))
        self.policy = np.zeros((self.env.time_limit, self.env.observation_space.n))

    def comp_expected_reward(self, t, future):
        # You can iterate over elements in the observation or state space using:
        # for s in self.env.observation_space: and for i in self.env.event_space,
        # respectively

        # This function should implement the computation of the expected reward of all
        # state possible at time t, given the expected reward "future" for all states in
        # time t+1
        states_t = np.empty((self.env.num_tickets + 1))
        policy_t = np.empty((self.env.num_tickets + 1))
        for s in self.env.observation_space:
            values = []
            for a in self.env.action_space:
                value = 0
                for i in self.env.event_space:
                    if not (s == 0 and i == self.env.event_space.h):
                        value += self.env.get_event_p(i, a, s, t) * (self.env.get_reward(i, a, s, t) + future[s - i])
                values.append(value)
            states_t[s] = max(values)
            policy_t[s] = values.index(max(values))        
        return states_t, policy_t

    def comp_expected_final_reward(self):
        # This should compute the expected reward of all final states.
        states_T = np.arange(0, self.env.num_tickets + 1)
        return states_T * self.env.f

    def solve(self):
        self.reset()
        # This should use the two functions above to fill the value and policy
        # matrix with the correct values. Policy maps each state at time t to an action,
        # Value maps each state at time t to its respective value.
        t = self.env.time_limit
        self.value[t] = self.comp_expected_final_reward()
        while t > 0:
            t -= 1
            self.value[t], self.policy[t] = self.comp_expected_reward(t, self.value[t+1])