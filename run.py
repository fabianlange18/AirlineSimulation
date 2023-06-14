from airline_environment import AirlineEnvironment

from calculation.dp_calculation import calculate_perfect_policy
from calculation.adp_ql_calculation import adp_ql_calculation
from calculation.dl_training import dl_training

import numpy as np
import matplotlib.pyplot as plt

discrete_env = AirlineEnvironment(continuous_action_space=False)
cont_env = AirlineEnvironment(continuous_action_space=True)

perfect_policy, perfect_value, perfect_reward = calculate_perfect_policy(discrete_env)

episodes_array = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]

adp = []
ql  = []
dqn = []
a2c = []
ppo = []

for episodes in episodes_array:
    adp_policy, _, adp_reward = adp_ql_calculation('adp', discrete_env, episodes, perfect_policy, perfect_value)
    ql_policy, _, ql_reward = adp_ql_calculation('ql', discrete_env, episodes, perfect_policy, perfect_value)
    dqn_policy, dqn_reward = dl_training('dqn', discrete_env, episodes, compare_policy=perfect_policy)
    a2c_policy, a2c_reward = dl_training('a2c', discrete_env, episodes, compare_policy=perfect_policy)
    ppo_policy, ppo_reward = dl_training('ppo', discrete_env, episodes, compare_policy=perfect_policy)
    # dl_training('ddpg', cont_env, episodes, compare_policy=perfect_policy)
    # dl_training('sac', cont_env, episodes, compare_policy=perfect_policy)

    adp.append(adp_reward)
    ql.append(ql_reward)
    dqn.append(dqn_reward)
    a2c.append(a2c_reward)
    ppo.append(ppo_reward)


n = len(adp)
r = np.arange(n)
width = 0.5 / n

plt.bar(r, adp, width=width, label='ADP')
plt.bar(r + width, ql, width=width, label='QL')
plt.bar(r + 2 * width, dqn, width=width, label='DQN')
plt.bar(r + 3 * width, a2c, width=width, label='A2C')
plt.bar(r + 4 * width, ppo, width=width, label='PPO')

plt.axhline(y=perfect_reward, color='black', linestyle='--')

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Reward per Method and Training Steps")

plt.xticks(r + 2 * width, episodes_array)
plt.legend(loc='upper left')

plt.savefig(f'./plots/summary')