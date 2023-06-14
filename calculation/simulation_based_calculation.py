from models.choose import choose_model
from util.colormap import plot_colormap

def calculate_simulation_based(model_name, env, k_steps, compare_policy = None, compare_value = None):

    assert model_name in ['adp', 'ql'], "Model name must be one of adp or ql"
    model = choose_model(model_name, env)
    model.solve(k_steps)

    model_name = 'Tabular_ADP' if model_name == 'adp' else 'Q-Learning'

    print(f"\nApproximate Policy by {model_name} after {k_steps} steps:")
    print(model.policy)

    if compare_policy is not None:
        plot_colormap(model.policy, compare_policy, model_name, k_steps, title='Policy')
    
    if compare_value is not None:
        plot_colormap(model.value, compare_value, model_name, k_steps, title='Values')

    return model.policy, model.value