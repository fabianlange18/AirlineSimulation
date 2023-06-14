from airline_environment import AirlineEnvironment

from models.choose import choose_model
from calculation.dp_calculation import calculate_perfect_policy
from calculation.simulation_based_calculation import calculate_simulation_based
from calculation.dl_training import dl_training

from simulation import simulation_run

env = AirlineEnvironment(continuous_action_space=False)

perfect_policy, perfect_value = calculate_perfect_policy(env)

steps_array = [100, 500, 1000 , 5000, 10000, 50000]

for steps in steps_array:
    calculate_simulation_based('adp', env, steps, perfect_policy, perfect_value)
    calculate_simulation_based('ql', env, steps, perfect_policy, perfect_value)
    dl_training('dqn', env, steps, compare_policy=perfect_policy)
    # dl_training('ddpg', env, steps, compare_policy=perfect_policy)
    dl_training('a2c', env, steps, compare_policy=perfect_policy)
    # dl_training('sac', env, steps, compare_policy=perfect_policy)
    dl_training('ppo', env, steps, compare_policy=perfect_policy)