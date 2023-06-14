import numpy as np

from models.choose import choose_model
from util.colormap import plot_colormap
from util.possible_states import setup_possible_states_array
from simulation import simulation_run

def dl_training(model_name, env, episodes, wandb = False, compare_policy = None):

    steps = env.booking_time * episodes

    assert model_name in ['dqn', 'ddpg', 'a2c', 'sac', 'ppo'], "Model name must be one of dqn, ddpg, a2c, sac or ppo."
    model = choose_model(model_name, env)

    if wandb:
        wandb.init(
            project="Airline Ticket Simulation",
            sync_tensorboard=True,
            mode='online'
        )

    print(model_name)
    model.learn(steps, progress_bar=True)

    possible_states = setup_possible_states_array(env)

    policy = np.zeros(env.observation_space.nvec)

    for state in possible_states:
        policy[*state] = model.predict(state, deterministic=True)[0]

    if compare_policy is not None:
        plot_colormap(policy, compare_policy, model_name, episodes, title='Policy')

    simulation_run(policy, model_name, episodes)

    return policy