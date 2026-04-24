# %% [markdown]
# ## HuReTEx AC 0.01 (2026.04.24) - Artifact Clustering

# %%
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import Birch
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from interface import implements

# %%
from HuReTEx_ACI_0_01 import ArtifactClusteringInterface

# %%
class EntropySVD:

    def __init__(self, X):
        
        self.X_np = X
        self.n_samples, self.n_features = self.X_np.shape

        self.USE_DATA_CLEANING = False         # enable/disable preprocessing step
        self.USE_STANDARDIZATION = False       # scaling before SVD

        self.VAR_EPS = 1e-6
        self.N_BINS_HIST = 30
        self.PERCENTILE_EPS = 5
        self.variance_target = 0.90

    def column_entropy(self, col):
        
        if np.max(col) == np.min(col):
            return 0.0

        col_norm = (col - np.min(col)) / (np.max(col) - np.min(col))

        hist, _ = np.histogram(col_norm, bins=self.N_BINS_HIST)
        p = hist / np.sum(hist)
        p = p[p > 0]

        return -np.sum(p * np.log(p))
    
    def components_for_variance(self, cum_var, threshold):
        
        idx = np.searchsorted(cum_var, threshold)
        return min(idx + 1, len(cum_var))
    
    def calculate(self):

        col_entropy = np.array([self.column_entropy(self.X_np[:, i]) for i in range(self.X_np.shape[1])])

        ENT_EPS = np.percentile(col_entropy, self.PERCENTILE_EPS)
        low_entropy_cols = np.where(col_entropy < ENT_EPS)[0]

        # ============================================================
        # FEATURE FILTERING
        # ============================================================
        self.X_np = np.delete(self.X_np, low_entropy_cols, axis=1)
        
        # ============================================================
        # NORMALIZATION (OPTIONAL)
        # ============================================================
        if self.USE_STANDARDIZATION:
            scaler = StandardScaler()
            self.X_np = scaler.fit_transform(self.X_np)

        # ============================================================
        # SVD
        # ============================================================
        max_components = min(300, self.n_features, self.n_samples - 1)

        svd_test = TruncatedSVD(
            n_components=max_components,
            random_state=42
        )

        X_tmp = svd_test.fit_transform(self.X_np)

        cum_var = np.cumsum(svd_test.explained_variance_ratio_)

        # ============================================================
        # COMPONENT SELECTION
        # ============================================================
        for threshold in [0.85, 0.90, 0.95, 0.99]:
            comp = self.components_for_variance(cum_var, threshold)

        # ============================================================
        # FINAL COMPONENT SELECTION
        # ============================================================
        n_components_selected = self.components_for_variance(cum_var, self.variance_target)
        
        # ============================================================
        # DIMENSIONALITY REDUCTION
        # ============================================================
        X_reduced = X_tmp[:, :n_components_selected]

        return X_reduced
    
# %%
class BirchClustering(implements(ArtifactClusteringInterface)):

    def __init__(self, X):
        self.X = X

    def fast_silhouette(self, labels, sample_size=1000):

        # Use sampling for efficiency on large datasets
        if len(self.X) > sample_size:
            idx = np.random.choice(len(self.X), sample_size, replace=False)
            return silhouette_score(self.X[idx], labels[idx], metric='euclidean')
        else:
            return silhouette_score(self.X, labels, metric='euclidean')
        
    def generate_thresholds(self, sample_size=1000):
        
        n_samples = min(sample_size, len(self.X))

        # Random subset for distance estimation
        idx = np.random.choice(len(self.X), n_samples, replace=False)
        sample = self.X[idx]

        # Compute pairwise distances
        dists = pairwise_distances(sample)
        median_dist = np.median(dists)

        # Generate threshold scaling factors
        factors = np.linspace(0.2, 1.2, 8)

        return median_dist * factors
    
    def evaluate_k(self, labels):

        # Require at least 2 clusters for meaningful metrics
        if len(np.unique(labels)) < 2:
            return None

        try:
            sil = self.fast_silhouette(self.X, labels)
            db = davies_bouldin_score(self.X, labels)
            ch = calinski_harabasz_score(self.X, labels)
        except:
            return None

        return {
            "sil": sil,
            "db": db,
            "ch": ch
        }
    
    def auto_birch_table_micro(self, k_range=(2,8)):

        thresholds = self.generate_thresholds()

        best_score = -np.inf
        best_model = None
        best_labels = None
        best_t = None
        best_k = None

        # Best model metrics
        best_sil = None
        best_ch = None
        best_db = None

        for t in thresholds:

            # Suppress convergence warnings during clustering
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                base_model = Birch(threshold=t, n_clusters=None)
                base_model.fit(self.X)

            labels_micro = base_model.labels_
            n_subclusters = len(base_model.subcluster_centers_)

            # Evaluate microclusters
            if n_subclusters >= 2:
                scores_micro = self.evaluate_k(labels_micro)
            else:
                continue

            # Skip overly fragmented solutions
            if n_subclusters > 0.3 * len(self.X):
                continue

            table_results = []

            for k in range(k_range[0], k_range[1] + 1):

                if k >= n_subclusters:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = Birch(threshold=t, n_clusters=k)
                    labels = model.fit_predict(self.X)

                scores = self.evaluate_k(labels)

                if scores is None:
                    continue

                table_results.append((k, scores, labels, model))
            
            # Normalize metrics
            # sil_norm = MinMaxScaler().fit_transform(
            #     np.array([r[1]["sil"] for r in table_results]).reshape(-1, 1)
            # ).flatten()

            # ch_norm = MinMaxScaler().fit_transform(
            #     np.array([r[1]["ch"] for r in table_results]).reshape(-1, 1)
            # ).flatten()

            # db_norm = MinMaxScaler().fit_transform(
            #     np.array([r[1]["db"] for r in table_results]).reshape(-1, 1)
            # ).flatten()

            for i, (k, scores, labels, model) in enumerate(table_results):

                #score_sum = sil_norm[i] + ch_norm[i] + (1 - db_norm[i])
                score_sum = scores[0]

                # Update best model
                if score_sum > best_score:
                    best_score = score_sum
                    best_model = model
                    best_labels = labels
                    best_t = t
                    best_k = k

                    # Store best raw metrics
                    best_sil = scores["sil"]
                    best_ch = scores["ch"]
                    best_db = scores["db"]
            
            
            

        return best_model, best_labels, best_t, best_k, best_sil, best_ch, best_db
    
    def cluster(self):

        best_model, best_labels, best_t, best_k, best_sil, best_ch, best_db = self.auto_birch_table_micro(k_range=(2,20))

        # X_with_labels = pd.DataFrame(self.X)
        # X_with_labels["cluster BIRCH"] = best_labels
        
        print(best_k)

        self.labels_ = best_labels
