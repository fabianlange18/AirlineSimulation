# customer demand is price and time dependent
# based on 2 customer groups:
# family group: appears in the middle of the horizon and prefers cheap prices
# business group: appears at the very end of the horizon and is willing to pay more

# this function models the arrival intensity
def consumer_demand_timewise(x: int):
    # price_trend = [1, 1, 1.2, 1.5, 2, 1.5, 1.2, 1, 0.6, 0.3]
    # alternative with high demand from family arrival till end of booking period
    # and way more business customers than family customers
    return 0.002 * (x ** 4) - 0.037 * (x ** 3) + 0.2 * (x ** 2) - 0.256 * x + 0.1


# this function models the willingness to pay
def consumer_demand_price_wise(x: int):
    # time_trend = [0.1, 0.106, 0.12, 0.123, 0.125, 0.25, 0.375, 0.4375, 0.46875, 0.5, 0.46875, 0.4375, 0.375, 0.25, 0.2, 0.22, 0.23, 0.25, 0.3, 0.5]
    # alternative with way higher willingness to pay for business:
    # 0.003*x^4+-0.052*x^3+0.207*x^2+0.057*x+1
    return 0.003 * (x ** 4) - 0.051 * (x ** 3) + 0.203 * (x ** 2) + 0.065 * x + 1
