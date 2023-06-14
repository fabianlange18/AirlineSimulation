from models.choose import choose_model
from util.colormap import plot_colormap
from simulation import simulation_run

def adp_ql_calculation(model_name, env, episodes, compare_policy = None, compare_value = None):

    steps = env.booking_time * episodes

    assert model_name in ['adp', 'ql'], "Model name must be one of adp or ql."
    model = choose_model(model_name, env)
    model.solve(steps)

    model_name = 'Tabular_ADP' if model_name == 'adp' else 'Q-Learning'

    print(f"\nApproximate Policy by {model_name} after {episodes} episodes:")
    print(model.policy)

    if compare_policy is not None:
        plot_colormap(model.policy, compare_policy, model_name, episodes, title='Policy')
    
    if compare_value is not None:
        plot_colormap(model.value, compare_value, model_name, episodes, title='Values')

    simulation_run(model.policy, model_name, episodes)

    return model.policy, model.value