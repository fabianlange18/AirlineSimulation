from airline_environment import AirlineEnvironment

from calculation.dp_calculation import calculate_perfect_policy
from calculation.adp_ql_calculation import adp_ql_calculation
from calculation.dl_training import dl_training

from simulation import simulation_run

env = AirlineEnvironment(continuous_action_space=False)

perfect_policy, perfect_value = calculate_perfect_policy(env)
simulation_run(perfect_policy, 'Optimal Solution', '1')

episodes_array = [100 , 500, 1000, 5000]

for episodes in episodes_array:
    adp_policy, _ = adp_ql_calculation('adp', env, episodes, perfect_policy, perfect_value)
    ql_policy, _ = adp_ql_calculation('ql', env, episodes, perfect_policy, perfect_value)
    dqn_policy = dl_training('dqn', env, episodes, compare_policy=perfect_policy)
    # dl_training('ddpg', env, episodes, compare_policy=perfect_policy)
    a2c_policy = dl_training('a2c', env, episodes, compare_policy=perfect_policy)
    # dl_training('sac', env, episodes, compare_policy=perfect_policy)
    ppo_policy = dl_training('ppo', env, episodes, compare_policy=perfect_policy)
