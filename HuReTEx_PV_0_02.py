# %% [markdown]
# ## HuReTEx PV 0.02 (2025.10.26)

# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
def generate_path_visualisation(file_name, path_info, filters_conv_1, filters_conv_2, activations, artifact_clusters):

    plt.rcParams.update({'font.size': 14})

    source_cluster_indexes = path_info.loc[0, 'source'].split('_')
    target_cluster_indexes = path_info.loc[0, 'target'].split('_')

    # Determine the maximum number of columns (i.e., filters) for a unified subplot layout
    max_filters = max(len(filters_conv_1), len(filters_conv_2))

    # 4 rows: 2 per convolutional layer (one for activation maps, one for cluster histograms)
    fig, axes = plt.subplots(4, max_filters, figsize=(max_filters * 4, 16))
    fig.suptitle("Average activations and distributions for conv1 and conv2 filters")

    max_cluster_count = max(artifact_clusters.apply(lambda x: x.value_counts().iloc[0]))

    # -------------------
    # Layer conv1 (layer 0)
    # -------------------
    for i in range(len(filters_conv_1)):
        
        j = filters_conv_1[i]
        
        try:
            source_idx = int(source_cluster_indexes[i])
        except IndexError:
            continue  # Missing index in the path

        activation_data = activations[0][artifact_clusters[f'l0_f{j}'] == source_idx, :, :, j]
        cluster_data = artifact_clusters.loc[artifact_clusters[f'l0_f{j}'] == source_idx, :]

        if activation_data.size > 0:
            avg_activation = np.mean(activation_data, axis=0)
            axes[0, i].imshow(avg_activation, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Conv1 Filter {j}, Cluster {source_idx}")

            bins = np.arange(0, 9 + 1.5) - 0.5
            axes[1, i].hist(cluster_data['p'], bins=bins)
            axes[1, i].set_ylim(top=max_cluster_count)
            axes[1, i].set_xticks(bins + 0.5)

    # -------------------
    # Layer conv2 (layer 1)
    # -------------------
    for i in range(len(filters_conv_2)):

        j = filters_conv_2[i]

        try:
            target_idx = int(target_cluster_indexes[i])
        except IndexError:
            axes[2, i].text(0.5, 0.5, "Missing index", ha='center', va='center')
            axes[2, i].axis('off')
            axes[3, i].axis('off')
            continue

        mask = artifact_clusters[f'l1_f{j}'] == target_idx
        num_samples = mask.sum()

        if num_samples > 0:
            activation_data = activations[1][mask, :, :, j]
            avg_activation = np.mean(activation_data, axis=0)

            axes[2, i].imshow(avg_activation, cmap='gray')
            axes[2, i].axis('off')
            axes[2, i].set_title(f"Conv2 Filter {j}, Cluster {target_idx}")

            cluster_data = artifact_clusters.loc[mask]
            bins = np.arange(0, 9 + 1.5) - 0.5
            axes[3, i].hist(cluster_data['p'], bins=bins)
            axes[3, i].set_ylim(top=max_cluster_count)
            axes[3, i].set_xticks(bins + 0.5)
        else:
            axes[2, i].text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
            axes[2, i].axis('off')
            axes[2, i].set_title(f"Conv2 Filter {i}, Cluster {target_idx} — brak danych")
            axes[3, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_name+'.png', bbox_inches='tight')
    plt.rcdefaults()
    plt.clf()


