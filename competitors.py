import numpy as np


class Competitor:

    def __init__(self, max_price, step_size, booking_time):

        self.max_price = max_price
        self.step_size = step_size
        self.booking_time = booking_time

    # fix price
    def fix_price(self, fix_price: int):
        return fix_price

    # undercutting (with barrier)
    def undercutting(self, agent_action, barrier=0):
        return max(barrier, agent_action - self.step_size)

    # only depending on own capacity
    def storager(self, comp_capacity):
        if comp_capacity > 0:
            return int(self.max_price / self.step_size) * int(self.max_price / 4)
        else:
            return self.max_price

    # mixed strategy depending on time and agent price
    def early_undercutting(self, agent_action, time):
        if time < self.booking_time / 2:
            return self.undercutting(agent_action, 0)
        else:
            return self.max_price

    def play_action(self, choice, time, agent_action, comp_capacity, fix_price, barrier=0):
        if 'fix price' in choice:
            return self.fix_price(fix_price)
        if 'undercut' in choice:
            return self.undercutting(agent_action, barrier)
        if 'storage' in choice:
            return self.storager(comp_capacity)
        if 'early undercut' in choice:
            return self.early_undercutting(agent_action, time)
