import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


# -------------------------------------------------------------
# Utility: Evaluate clustering quality
# -------------------------------------------------------------

def evaluate_clustering(X, labels):
    """Returns Silhouette Score, DB Index, CH Index."""
    if len(set(labels)) <= 1 or len(set(labels)) >= len(labels):
        # Invalid clustering (e.g. all points in one cluster)
        return None, None, None

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, db, ch


# -------------------------------------------------------------
# K-MEANS
# -------------------------------------------------------------

def run_kmeans(X, k):
    """
    Runs KMeans clustering and returns:
    - labels
    - model object
    - evaluation metrics
    """
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(X)

    sil, db, ch = evaluate_clustering(X, labels)

    print(f"\n[KMEANS] k = {k}")
    print(f"Silhouette Score: {sil}")
    print(f"Davies-Bouldin Index: {db}")
    print(f"Calinski-Harabasz Index: {ch}")

    return labels, model


# -------------------------------------------------------------
# HIERARCHICAL CLUSTERING
# -------------------------------------------------------------

def run_hierarchical(X, n_clusters, linkage="ward"):
    """
    Runs Agglomerative clustering.
    Returns labels, model placeholder, metrics.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    sil, db, ch = evaluate_clustering(X, labels)

    print(f"\n[HIERARCHICAL] clusters = {n_clusters}, linkage = {linkage}")
    print(f"Silhouette Score: {sil}")
    print(f"Davies-Bouldin Index: {db}")
    print(f"Calinski-Harabasz Index: {ch}")

    return labels, model


# -------------------------------------------------------------
# DBSCAN
# -------------------------------------------------------------

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering.
    Returns labels, model, metrics.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    sil, db, ch = evaluate_clustering(X, labels)

    print(f"\n[DBSCAN] eps = {eps}, min_samples = {min_samples}")
    print(f"Clusters found: {len(set(labels))} (including -1 = noise)")
    print(f"Silhouette Score: {sil}")
    print(f"Davies-Bouldin Index: {db}")
    print(f"Calinski-Harabasz Index: {ch}")

    return labels, model


# -------------------------------------------------------------
# PCA for Visualization (Step 7)
# -------------------------------------------------------------

def get_pca_2d_projection(X):
    """
    Returns 2D PCA projection of X.
    Used only for visualization (cluster plots).
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    return reduced
