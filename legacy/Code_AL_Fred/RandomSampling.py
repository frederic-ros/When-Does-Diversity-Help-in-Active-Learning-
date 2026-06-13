import torch
from utils.QueryStrategy import QueryStrategy

class RandomSampling(QueryStrategy):
    def __init__(self):
        pass

    def query(self, n_samples, X_pool, *args, **kwargs):
        selected_indices = torch.randperm(len(X_pool))[:n_samples]
        return selected_indices