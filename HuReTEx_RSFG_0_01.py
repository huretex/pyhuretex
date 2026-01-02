# %% [markdown]
# ## HuReTEx RSFG 0.01 (2025.03.13)

# %%
import numpy as np
import pandas as pd

# %%
def get_rsfg_df(sis):
    """
    Calculates a Rough Set Flow Graph (RSFG) for a given Sequential Information System (SIS).

    Arguments:
    - sis (DataFrame): sequential information system
    
    Returns:
    - rsfg_df: rough set flow graph in the form of a data frame
    """

    num_cols = sis.shape[1]

    rsfg_columns = ["level", "relation", "support", "certainty", "coverage", "strength"]

    rsfg_df = pd.DataFrame(columns=rsfg_columns)

    for level in range(num_cols - 1):
        
        edge_counts = {}
        source_counts = {}
        target_counts = {}
        total_possible = len(sis.iloc[:, level + 1])
        
        for _, row in sis.iterrows():
            
            source = str(row.iloc[level])
            target = str(row.iloc[level + 1])
            key = (source, target)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
            
        for (source, target), count in edge_counts.items():
            
            cer = count / source_counts[source] if source_counts[source] else 0
            cov = count / target_counts[target] if target_counts[target] else 0
            str_value = count / total_possible if total_possible else 0
            lev = f"{level}<>{level+1}"
            rel = f"{source}-->{target}"
            
            new_row = pd.DataFrame(data=np.array([[lev, rel, count, f"{cer:.6f}", f"{cov:.6f}", f"{str_value:.6f}"]]), columns=rsfg_columns)
            
            rsfg_df = pd.concat([rsfg_df, new_row], ignore_index=True)

        rsfg_df['confidence'] = 2*rsfg_df['certainty'].astype(float)*rsfg_df['coverage'].astype(float)/(rsfg_df['certainty'].astype(float)+rsfg_df['coverage'].astype(float))

    return rsfg_df

# %%
def get_rsfg_layers(rsfg_df):

    levels = rsfg_df['level'].unique()

    rsfg_layers = list()
    node_dicts = list()

    for level in levels:

        layer = rsfg_df.loc[rsfg_df['level']==level,:]
        layer[['source', 'target']] = layer['relation'].str.split('-->', expand=True)

        if len(node_dicts)==0:
            
            node_dict_1 = {key: i for i, key in enumerate(layer['source'].unique())}
            node_dicts.append(node_dict_1)

            layer['source_id'] = layer['source'].map(node_dict_1)

        else:

            layer['source_id'] = layer['source'].map(node_dicts[-1])        

        node_dict_2 = {key: i for i, key in enumerate(layer['target'].unique())}
        node_dicts.append(node_dict_2)

        layer['target_id'] = layer['target'].map(node_dict_2)
        
        layer = layer[['source_id', 'target_id', 'source', 'target', 'confidence']]
            
        rsfg_layers.append(layer)

    return (rsfg_layers, node_dicts)


