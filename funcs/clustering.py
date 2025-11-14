import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.decomposition import PCA


# -------------------------------------------------------------
# Utility: Evaluate clustering quality
# -------------------------------------------------------------
def evaluate_clustering(X, labels):
    """Returns Silhouette Score, DB Index, CH Index."""
    if len(set(labels)) <= 1 or len(set(labels)) >= len(labels):
        return None, None, None

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, db, ch


# -------------------------------------------------------------
# K-MEANS BASIC RUN
# -------------------------------------------------------------
def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(X)

    sil, db, ch = evaluate_clustering(X, labels)

    print(f"\n[KMEANS] k = {k}")
    print(f"Silhouette Score: {sil}")
    print(f"Davies-Bouldin Index: {db}")
    print(f"Calinski-Harabasz Index: {ch}")

    return labels, model


# -------------------------------------------------------------
# K-MEANS RANGE TESTING (for Elbow + Silhouette)
# -------------------------------------------------------------
def run_kmeans_range(X, k_values):
    inertias = []
    silhouettes = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)

        inertias.append(model.inertia_)

        if len(set(labels)) > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(None)

        print(f"[KMeans] k={k} | inertia={model.inertia_} | silhouette={silhouettes[-1]}")

    return inertias, silhouettes


# -------------------------------------------------------------
# Select Best k and Re-Run KMeans
# -------------------------------------------------------------
def run_kmeans_bestk(X, k):
    print(f"\nRunning final KMeans with k={k}")
    return run_kmeans(X, k)


# -------------------------------------------------------------
# PLOTS: Elbow Curve
# -------------------------------------------------------------
def plot_elbow_curve(k_values, inertias):
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, inertias, marker='o')
    plt.title("Elbow Curve (KMeans)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------
# PLOTS: Silhouette Scores
# -------------------------------------------------------------
def plot_silhouette_scores(k_values, silhouettes):
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, silhouettes, marker='o')
    plt.title("Silhouette Scores (KMeans)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------
# PCA for Visualization (Step 7)
# -------------------------------------------------------------
def get_pca_2d_projection(X):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    return reduced


# -------------------------------------------------------------
# PLOT PCA Clusters
# -------------------------------------------------------------
def plot_clusters_pca(X, labels, title):
    reduced = get_pca_2d_projection(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    plt.show()


# -------------------------------------------------------------
# HIERARCHICAL CLUSTERING
# -------------------------------------------------------------
def run_hierarchical(X, n_clusters, linkage="ward"):
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
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    sil, db, ch = evaluate_clustering(X, labels)

    print(f"\n[DBSCAN] eps = {eps}, min_samples = {min_samples}")
    print(f"Clusters found: {len(set(labels))} (including -1 = noise)")
    print(f"Silhouette Score: {sil}")
    print(f"Davies-Bouldin Index: {db}")
    print(f"Calinski-Harabasz Index: {ch}")

    return labels, model
