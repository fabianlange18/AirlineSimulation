from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment
from models.backwardinduction import BackwardInduction
from models.policyiteration import PolicyIteration
from models.approximatedp import ADP

from simulation import simulation_run

import wandb

env = AirlineEnvironment(continuous_action_space=False)

bi = BackwardInduction(env)

bi.solve()

print(bi.policy)
print(bi.value)

simulation_run(bi.policy)


# model = PPO(policy = 'MlpPolicy', env = env, gamma=0.99999, tensorboard_log='./logs')

# n_steps = 500000

# wandb.init(
#     project="Airline Ticket Simulation",
#     sync_tensorboard=True,
#     mode='online'
# )

# model.learn(n_steps, progress_bar=True)

# simulation_run(model)
