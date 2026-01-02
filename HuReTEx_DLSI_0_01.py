# %% [markdown]
# ## HuReTEx DLSI 0.01 (2025.03.13)

# %%
from interface import Interface

# %%
class DeepLearningSystemInterface(Interface):

    def train_model(self):
        pass

    def calculate_activations(self):
        pass

    def calculate_artifact_clusters(self):
        pass

    def get_sequential_information_system(self):
        pass


