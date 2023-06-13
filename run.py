from airline_environment import AirlineEnvironment

from models.choose import choose_model
from calculation.dp_calculation import calculate_perfect_policy
from calculation.simulation_based_calculation import calculate_simulation_based

from simulation import simulation_run

env = AirlineEnvironment(continuous_action_space=False)

perfect_policy, perfect_value = calculate_perfect_policy(env)

steps_array = [100, 500, 1000, 5000, 10000, 50000]

for steps in steps_array:
    approximate_policy, approximate_value = calculate_simulation_based('adp', env, steps, perfect_policy, perfect_value)
    approximate_policy, approximate_value = calculate_simulation_based('ql', env, steps, perfect_policy, perfect_value)

# model = choose_model('ppo', env)

# n_steps = 50

# # wandb.init(
# #     project="Airline Ticket Simulation",
# #     sync_tensorboard=True,
# #     mode='online'
# # )

# model.learn(n_steps, progress_bar=True)

# simulation_run(model)