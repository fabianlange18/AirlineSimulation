from models.estimator import Estimator

from calculation.dp_calculation import calculate_perfect_policy
from calculation.adp_ql_calculation import adp_ql_calculation
from calculation.dl_training import dl_training

from util.calculate_init_value import InitialValueCalculator

import csv
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

duopol = False

if not duopol:
    from airline_environment import AirlineEnvironment
else:
    from duopoly_environment import DuopolyEnvironment as AirlineEnvironment


discrete_env = AirlineEnvironment(continuous_action_space=False)
cont_env = AirlineEnvironment(continuous_action_space=True)

init_value_calculator = InitialValueCalculator(discrete_env)

perfect_policy, perfect_value, perfect_reward = calculate_perfect_policy(discrete_env, duopol=duopol)
assert abs(perfect_value[*discrete_env.initial_state] - init_value_calculator.calculate_initial_value(perfect_policy)) < 5

models_array = ['dp_est', 'adp', 'adp_est', 'ql', 'dqn', 'ddpg', 'td3', 'a2c', 'sac', 'ppo']
# models_array = ['dp_est', 'adp_est', 'ddpg', 'td3', 'sac', 'ppo'] # duopol methods
episodes_array = [1, 10, 100, 1000]

results = {model: {'r_means': [], 'r_std': [], 'v_means': [], 'v_std': []} for model in models_array}

n_runs = 5

for episodes in episodes_array:

    intermediate_results = {model: {'r': [], 'v': []} for model in models_array}
    
    for _ in range(n_runs):

        estimator = Estimator(discrete_env, n=episodes*discrete_env.booking_time, plot=True)

        for model_name in models_array:
            
            if model_name == 'dp_est':
                policy, value, reward = calculate_perfect_policy(discrete_env, estimator, just_result=True, duopol=duopol)
            
            elif model_name in ['adp', 'adp_est', 'ql']:
                policy, _, reward = adp_ql_calculation(model_name, discrete_env, episodes, estimator, perfect_policy, perfect_value, duopol=duopol)

            elif model_name in ['ql', 'dqn', 'a2c', 'ppo']:
                policy, reward = dl_training(model_name, discrete_env, episodes, compare_policy=perfect_policy, duopol=duopol)

            else:
                policy, reward = dl_training(model_name, cont_env, episodes, compare_policy=perfect_policy, duopol=duopol)

            discrete_env.reset()
            cont_env.reset()

            intermediate_results[model_name]['r'].append(reward)
            intermediate_results[model_name]['v'].append(init_value_calculator.calculate_initial_value(policy))

    for model_name in models_array:
        results[model_name]['r_means'].append(np.mean(intermediate_results[model_name]['r']))
        results[model_name]['r_std'].append(np.std(intermediate_results[model_name]['r']))
        results[model_name]['v_means'].append(np.mean(intermediate_results[model_name]['v']))
        results[model_name]['v_std'].append(np.std(intermediate_results[model_name]['v']))


    # CSV Datei Results
    # fields = ['model', 'r_means', 'r_std', 'v_means', 'v_std']
    with open('./plots/results.txt', 'a') as f:
        f.write(str(results))
        # w = csv.DictWriter(f, fields)
        # w.writeheader()
        # for key,val in sorted(results.items()):
        #     row = {'model': key}
        #     row.update(val)
        #     w.writerow(row)


n = len(episodes_array)
k = len(models_array)
r = np.arange(n)
width = 0.8 / len(models_array)

for i, model_name in enumerate(models_array):
    plt.bar(r + i * width, results[model_name]['r_means'], width=width, label = model_name, yerr = results[model_name]['r_std'])

plt.axhline(y=perfect_reward, color='black', linestyle=':')

plt.ylim(bottom=0)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title(f"Reward per Method and #Training Episodes (n={n_runs})")

plt.xticks(r + (k-1)/2 * width, episodes_array)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig(f'./plots/summary_r', bbox_inches="tight")
plt.close()

for i, model_name in enumerate(models_array):
    plt.bar(r + i * width, results[model_name]['v_means'], width=width, label = model_name, yerr = results[model_name]['v_std'])

plt.axhline(y=perfect_value[*discrete_env.initial_state], color='black', linestyle='--')

plt.ylim(bottom=0)
plt.xlabel("Episodes")
plt.ylabel("Expected Total Profit")
plt.title(f"Expected Total Profit per Method and #Training Episodes (n={n_runs})")

plt.xticks(r + (k-1)/2 * width, episodes_array)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig(f'./plots/summary_er', bbox_inches="tight")
plt.close()