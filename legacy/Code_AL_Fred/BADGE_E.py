from sklearn.cluster import kmeans_plusplus
import torch
import torch.nn.functional as F
from utils.QueryStrategy import QueryStrategy
from utils.ALModel import ALModel

class BADGE(QueryStrategy):
    def __init__(self, approximate=False):
        super(BADGE, self).__init__()
        self.approximate = approximate

    def query(self, model, X_pool, n_samples, device='cpu', *args, **kwargs):
        if self.approximate:
            return self.query_approx(model, X_pool, n_samples, device=device, *args, **kwargs)
        
        else:
            return self.query_sota(model, X_pool, n_samples, device=device, *args, **kwargs)

    def query_sota(self, model:ALModel, X_pool, n_samples, device='cpu', **kwargs):
        # Inutile pour toi car calcul des gradients exacts
        X_pool = X_pool.to(device)
        logits = model.predict_proba(X_pool)                             # [N, C]
        preds = torch.argmax(logits, dim=1)                 # [N]
        num_classes = logits.shape[1]
        
        # 2. One-hot des prédictions (differentiable)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, preds.unsqueeze(1), 1.0)
        
        # 3. Vrai calcul des gradients hypothétiques (cœur de BADGE)
        logits.requires_grad_(True)
        loss = F.cross_entropy(logits, preds, reduction='sum')
        grads = torch.autograd.grad(loss, logits)[0]        # [N, C] ← vrai gradient
        
        # 4. BADGE core : garder uniquement le gradient dans la direction prédite
        grad_embedding = grads * one_hot                    # [N, C]
        
        # 5. Amélioration SOTA obligatoire 2025 : concaténer la magnitude du gradient total
        grad_magnitude = grads.norm(p=2, dim=1, keepdim=True)   # [N, 1] ← capture l'incertitude
        grad_embedding = torch.cat([grad_embedding, grad_magnitude], dim=1)  # [N, C+1]
        
        # 6. Normalisation ℓ2 (absolument critique)
        grad_embedding = grad_embedding / (
            grad_embedding.norm(p=2, dim=1, keepdim=True) + 1e-8
        )
        
        # 7. Sélection k-means++ (la plus performante et rapide pour N ≤ 30k)
        grad_np = grad_embedding.detach().cpu().numpy()
        _, selected_idx = kmeans_plusplus(
            grad_np,
            n_clusters=n_samples,
            # n_local_trials=10,      # petit compromis vitesse/perf (10 est optimal)
            # random_state=42
        )
        
        return selected_idx.tolist()
    
    def query_approx(self, model, X_pool, n_samples, device='cpu', *args, **kwargs):
        # Check if model is an Ensemble or ALModel
        is_ensemble = hasattr(model, 'models') and hasattr(model, 'compute_avg_outputs')
        
        with torch.no_grad():
            if is_ensemble:
                avg_outputs = model.compute_avg_outputs(X_pool.to(device))
            else:
                avg_outputs = model.predict_proba(X_pool.to(device))
            
            num_classes = avg_outputs.shape[1]
            _, preds = torch.max(avg_outputs, 1)
            preds_onehot = F.one_hot(preds, num_classes=num_classes).float()
            # Approximate gradients
            avg_gradients = avg_outputs - preds_onehot  # Shape: (N, num_classes)
            avg_gradients = avg_gradients.cpu().numpy()

        # KMeans++ on gradients
        centers, indices = kmeans_plusplus(avg_gradients, n_samples)
        return indices

    