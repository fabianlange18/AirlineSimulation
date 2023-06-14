from models.choose import choose_model
from util.colormap import plot_colormap
from simulation import simulation_run

def adp_ql_calculation(model_name, env, episodes, compare_policy = None, compare_value = None, print_policy = False):

    steps = env.booking_time * episodes

    assert model_name in ['adp', 'ql'], "Model name must be one of adp or ql."
    model = choose_model(model_name, env)
    model.solve(steps)


    if compare_policy is not None:
        plot_colormap(model.policy, compare_policy, model_name, episodes, title='Policy')
    
    if compare_value is not None:
        plot_colormap(model.value, compare_value, model_name, episodes, title='Values')
    
    if print_policy:
        print(f"\nApproximate Policy by {model_name} after {episodes} episodes:")
        print(model.policy)

    reward = simulation_run(model.policy, model_name, episodes)

    return model.policy, model.value, reward