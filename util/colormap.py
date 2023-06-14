import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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