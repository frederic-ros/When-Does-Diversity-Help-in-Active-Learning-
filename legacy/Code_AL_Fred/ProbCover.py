from skactiveml.pool import ProbCover
from skactiveml.utils import unlabeled_indices, labeled_indices, MISSING_LABEL
from utils.QueryStrategy import QueryStrategy
from utils.ALModel import ALModel
import numpy as np

class ProbCoverSampling(QueryStrategy):
    def __init__(self):
        super(ProbCoverSampling, self).__init__()

    def query(self, model:ALModel, X_pool, n_samples, device='cpu', **kwargs):
        
        X = X_pool.numpy()
        y = np.full(shape=(X.shape[0]), fill_value=MISSING_LABEL)

        probcover = ProbCover()
        query_indices = probcover.query(X, y, batch_size=n_samples)

        return query_indices