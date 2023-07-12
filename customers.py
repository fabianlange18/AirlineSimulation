import numpy as np
import math

class Customers():

    def __init__(self, customers: list, max_price, booking_time):
        
        self.customers = customers
        self.max_price = max_price
        self.booking_time = booking_time
    
    
    # 1. Rational Customer
    def calculate_p_rational(self, a, timestep):
        return (1 - a / self.max_price) * (1 + timestep) / self.booking_time
    
    # 2. Family
    def calculate_p_family(self, a, timestep):
        time_factor = np.power(np.e, (- 1 / 2) * np.power(((timestep - self.booking_time / 2) / 1), 2))
        return (1 - a / self.max_price) * time_factor
    
    # 3. Business Customer
    def calculate_p_business(self, a, timestep):
        return (3. / 4.) / (timestep + 1) + 0.25
    
    # 4. Early Booking Customer
    def calculate_p_early_booking(self, a, timestep):
        return (1 - a / self.max_price) * (1 - timestep / self.booking_time)
    
    # 5. Party Customer
    def calculate_p_party(self, a, timestep):
        return (1 - a / self.max_price / 2) * (1 + timestep) / self.booking_time
    

    def calculate_p(self, a, timestep):
        p = 0
        counter = 0.0
        if 'rational' in self.customers:
            p+= self.calculate_p_rational(a, timestep)
            counter += 1
        if 'family' in self.customers:
            p+= self.calculate_p_family(a, timestep)
            counter += 1
        if 'business' in self.customers:
            p+= self.calculate_p_business(a, timestep)
            counter += 1
        if 'early_booking' in self.customers:
            p+= self.calculate_p_early_booking(a, timestep)
            counter += 1
        if 'party' in self.customers:
            p+= self.calculate_p_party(a, timestep)
            counter += 1
        p /= counter
        return np.where(p > 1, 1, p)
