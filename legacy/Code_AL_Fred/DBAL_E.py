import torch
import torch.nn.functional as F
from utils.QueryStrategy import QueryStrategy
from sklearn.cluster import KMeans
import numpy as np

class DBAL(QueryStrategy):
    
    def __init__(self, method='margin', dbal_factor=5, n_mc_samples=10):
        self.method = method
        self.dbal_factor = dbal_factor
        self.n_mc_samples = n_mc_samples

    def query(self, model, X_pool, n_samples, device='cpu', *args, **kwargs):
        
        # Compute uncertainty scores for all points in the pool
        if self.method == 'margin':
            probs = model.predict_proba(X_pool.to(device))
            top2_probs, _ = torch.topk(probs, 2, dim=1)
            margins = top2_probs[:, 0] - top2_probs[:, 1]
            uncertainties = 1 - margins
        elif self.method == 'entropy':
            probs = model.predict_proba(X_pool.to(device))
            log_probs = F.log_softmax(probs, dim=1)
            entropy = -torch.sum(probs * log_probs, dim=1)
            uncertainties = entropy
        else:
            raise ValueError("Unknown method")
        
        uncertain_indices = torch.topk(uncertainties, min(n_samples*self.dbal_factor, len(uncertainties))).indices

        # Do a weighted kmeans 
        kmeans = KMeans(n_clusters=n_samples, random_state=0)

        # Add 1e-8 to uncertainties to avoid zero weights in the kmeans clustering
        uncertainties = uncertainties + 1e-8

        kmeans.fit(X_pool.cpu()[uncertain_indices.cpu().numpy()], sample_weight=uncertainties[uncertain_indices].cpu().numpy())
        centers = kmeans.cluster_centers_
        indices = []
        for center in centers:
            center_tensor = torch.tensor(center, dtype=X_pool.dtype, device=device)
            dists = torch.norm(X_pool.to(device)[uncertain_indices.to(device)] - center_tensor, dim=1)
            closest_idx = torch.argmin(dists).item()
            indices.append(closest_idx)
        return uncertain_indices[indices].cpu().numpy()