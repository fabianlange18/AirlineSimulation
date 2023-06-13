from models.choose import choose_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


def plot_colormap(actual, compare_value, model_name, k_steps, title, print_labels = False):
    diff = np.abs(actual - compare_value)
    plt.imshow(diff, cm.get_cmap('RdBu').reversed())
    plt.colorbar()

    if print_labels:
        rows, cols = np.indices(diff.shape)

        for i, j in zip(rows.flatten(), cols.flatten()):
            plt.text(j, i, f"{diff[i, j]:.0f}", ha='center', va='center', color='white')

    plt.title(f'Deviation from optimal {title}')
    plt.savefig(f'./plots/{model_name}_{title}_{k_steps}')
    plt.close()