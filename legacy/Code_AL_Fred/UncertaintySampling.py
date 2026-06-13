import torch
import torch.nn.functional as F
from utils.QueryStrategy import QueryStrategy

class UncertaintySampling(QueryStrategy):
    def __init__(self, method='least_confident', n_mc_samples=10):
        self.n_mc_samples = n_mc_samples
        self.method = method

    def query(self, model, X_pool, n_samples, *args, **kwargs):
        
        with torch.no_grad():
            probs = model.predict_proba(X_pool)
            _, preds = torch.max(probs, 1)
            if self.method == 'least_confident':
                uncertainties = 1 - torch.max(probs, dim=1)[0]
            elif self.method == 'entropy':
                log_probs = F.log_softmax(probs, dim=1)
                entropy = -torch.sum(probs * log_probs, dim=1)
                uncertainties = entropy
            elif self.method == 'margin':
                top2_probs, _ = torch.topk(probs, 2, dim=1)
                margins = top2_probs[:, 0] - top2_probs[:, 1]
                uncertainties = 1 - margins
            else:
                raise ValueError("Unknown method")
        selected_indices = torch.topk(uncertainties, n_samples).indices
        return selected_indices.cpu()