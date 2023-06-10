# from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment
from models.backwardinduction import BackwardInduction
from models.policyiteration import PolicyIteration
from models.valueiteration import ValueIteration
from models.approximatedp import ADP

from simulation import simulation_run

import wandb
import numpy as np

env = AirlineEnvironment(continuous_action_space=False)

bi = BackwardInduction(env)
adp = ADP(env)
pi = PolicyIteration(env)
vi = ValueIteration(env)

k = 10000

bi.solve()
adp.solve(K=k)
pi.solve()
vi.solve()

assert(np.all(bi.policy == pi.policy))
assert(np.all(bi.policy == vi.policy))
assert(np.all(pi.policy == vi.policy))

print("Ideal Policy calculated by BI, PI or VI:")
print(bi.policy)

print(f"Approximate Policy by Tabular Forward Simulation after {k} steps:")
# Currently using zeros as initial policy and constant eps
print(adp.policy)

simulation_run(pi.policy)


# model = PPO(policy = 'MlpPolicy', env = env, gamma=0.99999, tensorboard_log='./logs')

# n_steps = 500000

# wandb.init(
#     project="Airline Ticket Simulation",
#     sync_tensorboard=True,
#     mode='online'
# )

# model.learn(n_steps, progress_bar=True)

# simulation_run(model)