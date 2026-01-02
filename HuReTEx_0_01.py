# %% [markdown]
# ## HuReTEx Main Notebook 0.01 (2025.11.05) - MNIST Dataset

# %% [markdown]
# #### Imports

# %%
from HuReTEx_DLS_0_01_MNIST import ConvolutionalSimpleMNIST

# %%
from HuReTEx_RSFG_0_01 import get_rsfg_df, get_rsfg_layers

# %%
from HuReTEx_PV_0_02 import generate_path_visualisation

# %%
from HuReTEx_GA_0_02 import get_best_confident_path

# %% [markdown]
# #### Deep Learning System - Unreadable Model

# %%
output_dir = './Results_MNIST'

# %%
dls = ConvolutionalSimpleMNIST()

# %%
dls.train_model()

# %% [markdown]
# #### Sequential Information System (SIS)

# %%
dls.calculate_activations()
dls.calculate_artifact_clusters()

# %%
filters_conv_1 = list(range(len(dls.filter_names_conv_1)))
print(filters_conv_1)
filters_conv_2 = list(range(len(dls.filter_names_conv_2)))
print(filters_conv_2)

# %% [markdown]
# #### Sequential Information System (SIS)

# %%
n_classes = 10

sis = dls.get_sequential_information_system()

print(sis)

# %% [markdown]
# #### Rough Set Flow Graph (RSFG) - Readable Twin

# %%
rsfg = get_rsfg_df(sis)
print(rsfg)

# %%
rsfg_layers, node_dicts = get_rsfg_layers(rsfg)
print(rsfg_layers)
print(node_dicts)

# %% [markdown]
# #### Best Paths (BPs)

# %%
varbounds = []

for d in node_dicts:

    varbounds.append([0,len(d)-1])

for c2 in range(n_classes):

    varbounds[3] = [c2,c2]

    best_path_info, best_solution = get_best_confident_path(rsfg_layers, node_dicts, dls.n_clusters_conv_1, varbounds)

    file_name = 'Source_'+best_path_info['source'][0]+'_target_'+best_path_info['target'][2]

    generate_path_visualisation(output_dir+'/'+file_name, best_path_info, filters_conv_1, filters_conv_2, dls.activations, dls.artifact_clusters)



