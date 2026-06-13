from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.QueryStrategy import QueryStrategy
from utils.ALModel import ALModel
import numpy as np

class TypiClustSampling(QueryStrategy):
    def __init__(self, neighbors=5):
        super(TypiClustSampling, self).__init__()
        self.neighbors = neighbors

    def query(self, model:ALModel, X_pool, n_samples, device='cpu', **kwargs):
        
        # Cluster the pool data
        kmeans = KMeans(n_clusters=n_samples, random_state=0)
        kmeans.fit(X_pool.cpu().numpy())

        # Fit a KNN model to find nearest neighbors to each embedding
        knn = NearestNeighbors(n_neighbors=self.neighbors)
        knn.fit(X_pool.cpu().numpy())

        # Compute sum of distances to nearest neighbors for each point 
        distances, _ = knn.kneighbors(X_pool.cpu().numpy())
        sum_distances = distances.sum(axis=1)

        typicalities = 1 / (sum_distances)

        # For each cluster, select the most typical point
        query_indices = []
        for cluster in range(n_samples):
            cluster_indices = np.where(kmeans.labels_ == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_typicalities = typicalities[cluster_indices]
            most_typical_index = cluster_indices[np.argmax(cluster_typicalities)]
            query_indices.append(most_typical_index)

        return query_indices