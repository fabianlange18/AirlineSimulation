from airline_environment import AirlineEnvironment
from models.backwardinduction import BackwardInduction
from models.approximatedp import ADP
from models.estimator import Estimator

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from simulation import simulation_run
from util.colormap import plot_colormap

env = AirlineEnvironment(False)

# # plot for demonstration purposes
estimator = Estimator(env, 500, plot=True)

n = np.arange(20, 100, step=10)
k = 5

results = {n_i: {'reward': [], 'mse': [], 'initial_value': []} for n_i in n}

for _ in trange(k):
    for n_i in n:

        estimator = Estimator(env, 500, plot=False)

        adp = ADP(env)
        adp_estimate = ADP(env, estimator=estimator)

        adp.solve(50, prog_bar=False)
        adp_estimate.solve(50, prog_bar=False)

        plot_colormap(adp_estimate.policy, adp.policy, model_name= "0_Estimation", episodes = n_i, title="Policy")
        reward = simulation_run(adp_estimate.policy, model_name="0_Estimation", episodes=n_i, plot = False)
        results[n_i]['reward'].append(reward)
        results[n_i]['mse'].append(estimator.mse)
        results[n_i]['initial_value'].append(adp_estimate.value[*env.initial_state])


plt.figure()
y = np.array([np.mean(results[n_i]['reward']) for n_i in n])
std = np.array([np.std(results[n_i]['reward']) for n_i in n])
plt.plot(n, y)
plt.fill_between(n, (y - std), (y + std), alpha=0.5)
plt.title(f"Rewards after {k} Simulations")
plt.show()

plt.figure()
y = np.array([np.mean(results[n_i]['reward']) for n_i in n])
std = np.array([np.std(results[n_i]['reward']) for n_i in n])
plt.plot(n, y)
plt.fill_between(n, (y - std), (y + std), alpha=0.5)
plt.title("Initial Value of the Policy")
plt.title("Initial Value of the Policy")
plt.show()

plt.figure()
y = np.array([np.mean(results[n_i]['mse']) for n_i in n])
std = np.array([np.std(results[n_i]['mse']) for n_i in n])
plt.plot(n, y)
plt.fill_between(n, (y - std), (y + std), alpha=0.5)
plt.title("Estimator MSE")
plt.show()