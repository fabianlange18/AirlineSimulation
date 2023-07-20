from models.choose import choose_model
from util.colormap import plot_colormap
from simulation import simulation_run

import numpy as np

def adp_ql_calculation(model_name, env, episodes, estimator=None, compare_policy = None, compare_value = None, print_policy = False):

    steps = env.booking_time * episodes

    assert model_name in ['adp', 'adp_est', 'ql'], "Model name must be one of adp or ql."
    if model_name == 'adp_est':
        assert estimator
    
    model = choose_model(model_name, env, estimator)
    model.solve(steps)


    if compare_policy is not None:
        plot_colormap(model.policy, compare_policy, model_name, episodes, title='Policy')
    
    if compare_value is not None:
        plot_colormap(model.value, compare_value, model_name, episodes, title='Values')
    
    if print_policy:
        print(f"\nApproximate Policy by {model_name} after {episodes} episodes:")
        print(model.policy)

    rewards = []
    
    for _ in range(100):
        rewards.append(simulation_run(model.policy, model_name, episodes))
    
    reward = np.mean(rewards)

    return model.policy, model.value, reward