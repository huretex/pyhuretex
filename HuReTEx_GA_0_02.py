# %% [markdown]
# ## HuReTEx GA 0.02 (2025.10.23)

# %%
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from functools import reduce

# %%
def min_tnorm(a, b):
    
    return min(a, b)

def product_tnorm(a, b):
    
    return a * b

def lukasiewicz_tnorm(a, b):
    
    return max(0, a + b - 1)

def fodor_tnorm(a, b):

    if a + b > 1:

        return min(a, b)
    
    else:

        return 0

def drastic_tnorm(a, b):

    if max(a, b) == 1:

        return min(a, b)
    
    else:

        return 0
    
def einstein_tnorm(a, b):

    return (a * b) / (2 - (a + b - a * b))


def max_snorm(a, b):
    
    return max(a, b)

def probabilistic_snorm(a, b):
    
    return a + b - a * b

def lukasiewicz_snorm(a, b):
    
    return min(1, a + b)

def fodor_snorm(a, b):

    if a + b < 1:

        return max(a, b)
    
    else:

        return 1

def drastic_snorm(a, b):

    if min(a, b) == 0:

        return max(a, b)
    
    else:

        return 1
    
def einstein_snorm(a, b):

    return (a + b) / (1  + a * b)

# %%

def apply_tnorm_vector(values, tnorm_func):
    
    return reduce(tnorm_func, values)


# %%
def fitness(x, rsfg_layers):

    confidences = list()

    for i in range(len(x)-1):

        layer = rsfg_layers[i]

        confidence = layer.loc[(layer['source_id']==x[i]) & (layer['target_id']==x[i+1]), ['confidence']]

        if len(confidence)==0:
            
            confidences.append(0.0)

        else:  
            
            confidences.append(confidence['confidence'].values[0])

    assessment = apply_tnorm_vector(np.array(confidences), einstein_tnorm) 

    return -assessment  

# %%
def get_best_confident_path(rsfg_layers, node_dicts, n_clusters_conv_1, varbounds):
    
    max_num_iteration = 10000

    parameters = {'max_num_iteration': max_num_iteration,
              'population_size': 100,
              'parents_portion': 0.3,
              'mutation_probability': 0.1,
              'crossover_probability': 0.5,
              'elit_ratio': 0.01,
              'crossover_type': 'uniform',
              'max_iteration_without_improv': int(0.5*max_num_iteration)}


    gen_alg = ga(function=lambda x: fitness(x, rsfg_layers), dimension=len(varbounds), variable_type='int', variable_boundaries=np.array(varbounds), algorithm_parameters=parameters)

    gen_alg.run()

    best_path = gen_alg.output_dict['variable']

    best_path_info = pd.DataFrame(columns=['confidence', 'source', 'target'])

    for i in range(len(best_path)-1):

            layer = rsfg_layers[i]

            path_info = layer.loc[(layer['source_id']==best_path[i]) & (layer['target_id']==best_path[i+1]), ['confidence', 'source', 'target']]

            if len(path_info) == 0:

                sources = layer.loc[(layer['source_id']==best_path[i]), ['source']].reset_index(drop=True)
                targets = layer.loc[(layer['target_id']==best_path[i+1]), ['target']].reset_index(drop=True)

                source = sources.loc[0, 'source']
                target = targets.loc[0, 'target']
                
                best_path_info = pd.concat([best_path_info.dropna(axis=1, how='all') , pd.DataFrame({'confidence': [0.0], 'source': [source], 'target': [target]})], ignore_index=True)

            else:
                 
                best_path_info = pd.concat([best_path_info.dropna(axis=1, how='all') , path_info], ignore_index=True)

    best_solution = -gen_alg.output_dict['function']

    return (best_path_info, best_solution)


