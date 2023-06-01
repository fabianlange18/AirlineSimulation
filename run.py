from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment

from simulation import simulation_run

import wandb

env = AirlineEnvironment()

# model = PPO('MlpPolicy', env, gamma=0.99999, tensorboard_log='./logs')

# n_steps = 500000

# wandb.init(
#     project="Airline Ticket Simulation",
#     sync_tensorboard=True,
#     mode='disabled'
# )

# model.learn(n_steps, progress_bar=True)

# simulation_run(model)

from models.policyiteration import PolicyIteration

PolicyIteration(env).solve()