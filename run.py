from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment

from models.backwardinduction import BackwardInduction
from models.policyiteration import PolicyIteration
from models.valueiteration import ValueIteration

from models.approximatedp import ADP
from models.qlearning import QLearning

from simulation import simulation_run

import wandb
import numpy as np

env = AirlineEnvironment(continuous_action_space=False)

bi = BackwardInduction(env)
pi = PolicyIteration(env)
vi = ValueIteration(env)

bi.solve()
pi.solve()
vi.solve()

assert(np.all(bi.policy == pi.policy))
assert(np.all(bi.policy == vi.policy))
assert(np.all(pi.policy == vi.policy))

print("\nPerfect Policy calculated by BI, PI and VI:")
print(bi.policy)
simulation_run(bi.policy)


adp = ADP(env)
ql = QLearning(env)

k = 1
adp.solve(K=k)
ql.solve(steps=k)

print(f"\nApproximate Policy by Tabular Forward Simulation after {k} steps:")
# Currently using zeros as initial policy and constant eps
print(adp.policy)

print(f"\nApproximate Policy by Q-Learning after {k} steps:")
# Currently using zeros as initial policy and constant eps
print(ql.policy)



model = PPO(policy = 'MlpPolicy', env = env, gamma=0.99999, tensorboard_log='./logs')

n_steps = 50

# wandb.init(
#     project="Airline Ticket Simulation",
#     sync_tensorboard=True,
#     mode='online'
# )

model.learn(n_steps, progress_bar=True)

simulation_run(model)