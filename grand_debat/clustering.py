from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compute_and_get_kmeans_clusters(embeddings_df, number_of_clusters):
    km = KMeans(n_clusters=number_of_clusters, random_state=123).fit(np.stack(embeddings_df.loc[:, "embeddings"].values))
    embeddings_df['cluster'] = km.predict(np.stack(embeddings_df.loc[:, "embeddings"].values))
    return embeddings_df


def compute_and_get_hierarchical_clusters(embeddings_df, number_of_clusters, output_path=None):
    Z = linkage(np.stack(embeddings_df.loc[:, "embeddings"].values), method="ward")

    if output_path is not None:
        # Plot dendrogram
        fig = plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        fig.savefig(output_path, dpi=fig.dpi)

    embeddings_df['cluster'] = fcluster(Z, number_of_clusters, criterion='maxclust')
    return embeddings_df
