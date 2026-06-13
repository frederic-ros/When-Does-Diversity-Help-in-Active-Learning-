import torch
from utils.QueryStrategy import QueryStrategy
from sklearn.cluster import kmeans_plusplus


class Coreset(QueryStrategy):
    def __init__(self, strategy='kmeans++'):
        self.strategy = strategy

    def query(self, X_pool, n_samples, *args, **kwargs):
        if self.strategy == 'greedy':
            return self.query_greedy(X_pool, n_samples, *args, **kwargs)
        elif self.strategy == 'kmeans++':
            centers, indices = kmeans_plusplus(X_pool.cpu().numpy(), min(n_samples, len(X_pool)))
            return indices
        else:
            raise ValueError("Unknown coreset strategy")

    def query_greedy(self, X_pool, n_samples, *args, **kwargs):
        # Coreset selection using set k-center Greedy
        selected_indices = []
        # Use broadcasting to compute distances
        distances = torch.cdist(X_pool, X_pool)
        min_distances = torch.full((len(X_pool),), float('inf'))
        for _ in range(n_samples):
            if len(selected_indices) == 0:
                idx = torch.randint(0, len(X_pool), (1,)).item()
            else:
                idx = torch.argmax(min_distances).item()
            selected_indices.append(idx)
            min_distances = torch.min(min_distances, distances[idx])
        return selected_indices