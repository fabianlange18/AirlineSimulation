from airline_environment import AirlineEnvironment

from calculation.dp_calculation import calculate_perfect_policy
from calculation.adp_ql_calculation import adp_ql_calculation
from calculation.dl_training import dl_training

from util.calculate_init_value import InitialValueCalculator

import numpy as np
import matplotlib.pyplot as plt

discrete_env = AirlineEnvironment(continuous_action_space=False)
cont_env = AirlineEnvironment(continuous_action_space=True)

perfect_policy, perfect_value, perfect_reward = calculate_perfect_policy(discrete_env)


init_value_calculator = InitialValueCalculator(discrete_env)

models_array = ['adp', 'ql', 'dqn', 'a2c', 'td3', 'ppo', 'ddpg', 'sac']
episodes_array = [1, 5, 10, 15] #, 100, 500, 1000, 5000, 10000, 50000]

results = {model: {'r_means': [], 'r_std': [], 'v_means': [], 'v_std': []} for model in models_array}

n_runs = 2

for episodes in episodes_array:

    intermediate_results = {model: {'r': [], 'v': []} for model in models_array}
    
    for _ in range(n_runs):

        for model_name in models_array:

            if model_name in ['adp', 'ql']:

                policy, _, reward = adp_ql_calculation(model_name, discrete_env, episodes, perfect_policy, perfect_value)

            elif model_name in ['ql', 'dqn', 'a2c', 'ppo']:

                policy, reward = dl_training(model_name, discrete_env, episodes, compare_policy=perfect_policy)

            else:

                policy, reward = dl_training(model_name, cont_env, episodes, compare_policy=perfect_policy)

            intermediate_results[model_name]['r'].append(reward)
            intermediate_results[model_name]['v'].append(init_value_calculator.calculate_initial_value(policy))

    for model_name in models_array:
        results[model_name]['r_means'].append(np.mean(intermediate_results[model_name]['r']))
        results[model_name]['r_std'].append(np.std(intermediate_results[model_name]['r']))
        results[model_name]['v_means'].append(np.mean(intermediate_results[model_name]['v']))
        results[model_name]['v_std'].append(np.std(intermediate_results[model_name]['v']))


n = len(episodes_array)
k = len(models_array)
r = np.arange(n)
width = 0.8 / len(models_array)

for i, model_name in enumerate(models_array):
    plt.bar(r + i * width, results[model_name]['r_means'], width=width, label = model_name, yerr = results[model_name]['r_std'])

plt.axhline(y=perfect_reward, color='black', linestyle='--')

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward per Method and #Training Episodes")

plt.xticks(r + (k-1)/2 * width, episodes_array)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig(f'./plots/summary_r', bbox_inches="tight")
plt.close()

for i, model_name in enumerate(models_array):
    plt.bar(r + i * width, results[model_name]['v_means'], width=width, label = model_name, yerr = results[model_name]['v_std'])

plt.axhline(y=perfect_value[*discrete_env.initial_state], color='black', linestyle='--')

plt.xlabel("Episodes")
plt.ylabel("Initial Values")
plt.title("Initial Values per Method and #Training Episodes")

plt.xticks(r + (k-1)/2 * width, episodes_array)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig(f'./plots/summary_v', bbox_inches="tight")
plt.close()