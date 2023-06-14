import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_colormap(actual, compare_value, model_name, episodes, title, print_labels = False):
    diff = np.abs(actual - compare_value)
    plt.imshow(diff, cm.get_cmap('RdBu').reversed())
    plt.colorbar()

    if print_labels:
        rows, cols = np.indices(diff.shape)

        for i, j in zip(rows.flatten(), cols.flatten()):
            plt.text(j, i, f"{diff[i, j]:.0f}", ha='center', va='center', color='white')

    plt.title(f'Deviation from optimal {title}')
    plt.xlabel(xlabel="flight capacity")
    plt.ylabel(ylabel="booking time")
    plt.savefig(f'./plots/colormaps/{model_name}_{title}_{episodes}')
    plt.close()