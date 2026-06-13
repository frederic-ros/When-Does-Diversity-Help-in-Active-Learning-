# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:54:56 2026

@author: frederic.ros
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy

from typing import Tuple, List, Optional


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)

    if proba.ndim != 2 or proba.shape[1] < 2:
        p = np.clip(proba.reshape(-1), 0.0, 1.0)
        return 1.0 - p

    sorted_p = -np.sort(-proba, axis=1)
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    return 1.0 - margin


def entropy_uncertainty(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    p = np.clip(proba, eps, 1.0)

    if p.ndim != 2:
        p = p.reshape(-1, 1)

    h = -np.sum(p * np.log(p), axis=1)
    c = max(2, p.shape[1])
    return h / (np.log(c) + eps)


def _safe_minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / ((mx - mn) + eps)


def _safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0

    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < eps or sy < eps:
        return 0.0

    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def _safe_predict_proba(model, X: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    Robust predict_proba with remapping on the full class space when needed.
    """
    proba = model.predict_proba(X)

    if n_classes is None:
        return np.asarray(proba, dtype=float)

    out = np.zeros((len(X), n_classes), dtype=float)
    classes_seen = np.asarray(model.classes_, dtype=int)
    out[:, classes_seen] = proba

    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return out / row_sum


@register("ActivePseudoLabelV4")
class ActivePseudoLabelV4(QueryStrategy):
    """
    Adaptive version of V4.
    """

    def __init__(
        self,
        *,
        mix_entropy: bool = True,
        entropy_weight: float = 0.20,

        adaptive_k: bool = True,
        k_neighbors: int = 10,
        k_min: int = 5,
        k_max: int = 25,
        k_mode: str = "sqrt",

        lambda_prop: float = 0.35,
        alpha_decay: float = 2.0,
        adaptive_alpha: bool = True,
        alpha_min_factor: float = 0.60,
        alpha_max_factor: float = 1.40,

        min_source_score: float = 0.12,
        use_local_support: bool = True,

        adaptive_lambda: bool = True,
        credible_errors_tau: float = 8.0,
        use_quality_adaptation: bool = True,
        use_shape_adaptation: bool = True,

        use_gate: bool = True,
        gate_std_threshold: float = 0.25,
        gate_corr_threshold: float = 0.90,
        gate_soft: bool = True,

        weighted_kmeans: bool = True,
        weight_power: float = 1.0,
        pool_multiplier: float = 1.5,
        random_state: int = 0,

        normalize_once_at_end: bool = True,
        use_safe_proba_mapping: bool = False,
        eps: float = 1e-8,

        # --- Honest (cross-validated) error sources ------------------
        # A memorising backbone (e.g. RandomForest) has zero training
        # error, so _source_scores would never find propagation
        # sources. honest_errors replaces the memorised proba_labeled
        # by an out-of-fold cross_val_predict estimate. ON by default.
        honest_errors: bool = True,
        honest_cv_max: int = 3,
        honest_min_cv: int = 2,
        honest_random_state: int = 0,
    ):
        self.honest_errors = bool(honest_errors)
        self.honest_cv_max = int(honest_cv_max)
        self.honest_min_cv = int(honest_min_cv)
        self.honest_random_state = int(honest_random_state)
        self._honest_last_cv = 0
        self._honest_last_active = False
        self._honest_model_ref = None

        self.mix_entropy = bool(mix_entropy)
        self.entropy_weight = float(entropy_weight)

        self.adaptive_k = bool(adaptive_k)
        self.k_neighbors = int(k_neighbors)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_mode = str(k_mode)

        self.lambda_prop = float(lambda_prop)
        self.alpha_decay = float(alpha_decay)
        self.adaptive_alpha = bool(adaptive_alpha)
        self.alpha_min_factor = float(alpha_min_factor)
        self.alpha_max_factor = float(alpha_max_factor)

        self.min_source_score = float(min_source_score)
        self.use_local_support = bool(use_local_support)

        self.adaptive_lambda = bool(adaptive_lambda)
        self.credible_errors_tau = float(credible_errors_tau)
        self.use_quality_adaptation = bool(use_quality_adaptation)
        self.use_shape_adaptation = bool(use_shape_adaptation)

        self.use_gate = bool(use_gate)
        self.gate_std_threshold = float(gate_std_threshold)
        self.gate_corr_threshold = float(gate_corr_threshold)
        self.gate_soft = bool(gate_soft)

        self.weighted_kmeans = bool(weighted_kmeans)
        self.weight_power = float(weight_power)
        self.pool_multiplier = float(pool_multiplier)
        self.random_state = int(random_state)

        self.normalize_once_at_end = bool(normalize_once_at_end)
        self.use_safe_proba_mapping = bool(use_safe_proba_mapping)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # Adaptive controls
    # ------------------------------------------------------------------

    def _effective_k(self, n_u: int) -> int:
        if n_u <= 0:
            return 1

        if not self.adaptive_k:
            return int(np.clip(self.k_neighbors, 1, n_u))

        if self.k_mode == "log":
            k_eff = int(np.round(2.0 * np.log2(n_u + 1.0)))
        else:
            k_eff = int(np.round(np.sqrt(n_u)))

        k_eff = int(np.clip(k_eff, self.k_min, self.k_max))
        k_eff = int(np.clip(k_eff, 1, n_u))
        return k_eff

    def _adaptive_alpha_from_distances(self, distances: np.ndarray) -> float:
        base_alpha = float(self.alpha_decay)

        if (not self.adaptive_alpha) or distances.size == 0:
            return base_alpha

        mu = float(np.mean(distances)) + self.eps
        sigma = float(np.std(distances))
        cv = sigma / mu

        factor = 1.0 / (1.0 + cv)
        factor = float(np.clip(factor, self.alpha_min_factor, self.alpha_max_factor))
        return base_alpha * factor

    # ------------------------------------------------------------------
    # Base uncertainty
    # ------------------------------------------------------------------

    def _base_uncertainty(self, proba_u: np.ndarray) -> np.ndarray:
        u_margin = margin_uncertainty(proba_u)
        if not self.mix_entropy:
            return u_margin

        u_entropy = entropy_uncertainty(proba_u, eps=self.eps)
        w = float(np.clip(self.entropy_weight, 0.0, 1.0))
        return (1.0 - w) * u_margin + w * u_entropy

    # ------------------------------------------------------------------
    # Source scoring
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _honest_proba_labeled(
        self,
        model,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        proba_labeled_legacy: np.ndarray,
        classes_seen: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Out-of-fold proba estimate (backbone-agnostic) so that
        propagation sources can be detected even with a perfectly
        memorising classifier. Falls back to the legacy proba when an
        honest estimate cannot be built safely (cold-start, a class
        with too few samples, CV failure).
        """
        self._honest_last_active = False
        self._honest_last_cv = 0

        if not self.honest_errors or model is None:
            return proba_labeled_legacy

        n_l = len(X_labeled)
        if n_l == 0:
            return proba_labeled_legacy

        y_labeled = np.asarray(y_labeled)
        uniq, counts = np.unique(y_labeled, return_counts=True)
        if uniq.size < 2:
            return proba_labeled_legacy

        min_per_class = int(counts.min())
        cv = min(int(self.honest_cv_max), min_per_class)
        if cv < int(self.honest_min_cv):
            return proba_labeled_legacy

        if classes_seen is None:
            classes_seen = np.unique(y_labeled)
        classes_seen = np.asarray(classes_seen)
        col_of = {int(c): j for j, c in enumerate(classes_seen)}

        try:
            skf = StratifiedKFold(
                n_splits=cv, shuffle=True,
                random_state=self.honest_random_state,
            )
            est = clone(model)

            if hasattr(est, "predict_proba"):
                oof = cross_val_predict(
                    est, X_labeled, y_labeled,
                    cv=skf, method="predict_proba",
                )
                fitted_classes = np.unique(y_labeled)
                proba = np.zeros((n_l, len(classes_seen)), dtype=float)
                for src_col, c in enumerate(fitted_classes):
                    c = int(c)
                    if c in col_of:
                        proba[:, col_of[c]] = oof[:, src_col]
            else:
                oof_lab = cross_val_predict(
                    est, X_labeled, y_labeled, cv=skf, method="predict",
                )
                proba = np.zeros((n_l, len(classes_seen)), dtype=float)
                for row, c in enumerate(oof_lab):
                    c = int(c)
                    if c in col_of:
                        proba[row, col_of[c]] = 1.0

            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0.0] = 1.0
            proba = proba / row_sums

            self._honest_last_active = True
            self._honest_last_cv = cv
            return proba

        except Exception as e:
            print(f"[honest_errors] CV failed ({e!r}); using legacy proba.")
            return proba_labeled_legacy

    def _source_scores(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        proba_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        proba_unlabeled: np.ndarray,
        k_eff: int,
        classes_seen: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper: substitute an honest proba_labeled, then delegate."""
        if self.honest_errors:
            proba_labeled = self._honest_proba_labeled(
                model=getattr(self, "_honest_model_ref", None),
                X_labeled=X_labeled,
                y_labeled=y_labeled,
                proba_labeled_legacy=proba_labeled,
                classes_seen=classes_seen,
            )
        return self._source_scores_raw(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            proba_labeled=proba_labeled,
            X_unlabeled=X_unlabeled,
            proba_unlabeled=proba_unlabeled,
            k_eff=k_eff,
            classes_seen=classes_seen,
        )

    def _source_scores_raw(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        proba_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        proba_unlabeled: np.ndarray,
        k_eff: int,
        classes_seen: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score only actual labeled errors.

        score_i =
            confident_wrong_i * local_uncertainty_i * local_true_support_i
        """
        n_l = len(X_labeled)
        n_u = len(X_unlabeled)

        if n_l == 0 or n_u == 0:
            return np.array([], dtype=float), np.array([], dtype=int)

        n_classes = proba_labeled.shape[1] if proba_labeled.ndim == 2 else 1

        if classes_seen is None:
            classes_seen = np.arange(n_classes, dtype=int)
        else:
            classes_seen = np.asarray(classes_seen, dtype=int)

        class_to_col = {int(c): j for j, c in enumerate(classes_seen)}

        # IMPORTANT: convert argmax columns -> real labels
        pred_l = classes_seen[np.argmax(proba_labeled, axis=1)]
        err_mask = pred_l != y_labeled
        err_indices = np.where(err_mask)[0]

        if err_indices.size == 0:
            return np.array([], dtype=float), np.array([], dtype=int)

        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X_unlabeled)
        U_u = self._base_uncertainty(proba_unlabeled)

        q_err = np.zeros(err_indices.size, dtype=float)

        for j, i in enumerate(err_indices):
            x_i = X_labeled[i]
            y_i = int(y_labeled[i])
            pred_i = int(pred_l[i])

            distances, indices = nbrs.kneighbors(x_i.reshape(1, -1))
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)

            w = np.exp(-0.5 * (distances / (np.mean(distances) + self.eps)) ** 2)
            w_sum = float(np.sum(w)) + self.eps

            if n_classes > 1 and y_i in class_to_col:
                col_true = class_to_col[y_i]
                p_true = float(np.clip(proba_labeled[i, col_true], 0.0, 1.0))
            else:
                p_true = 0.0 if n_classes > 1 else 1.0

            if n_classes > 1 and pred_i in class_to_col:
                col_pred = class_to_col[pred_i]
                p_pred = float(np.clip(proba_labeled[i, col_pred], 0.0, 1.0))
            else:
                p_pred = 0.0 if n_classes > 1 else 1.0

            confident_wrong = max(0.0, p_pred - p_true)
            local_uncertainty = float(np.sum(w * U_u[indices]) / w_sum)

            if self.use_local_support and n_classes > 1 and y_i in class_to_col:
                col_true = class_to_col[y_i]
                local_true_support = float(np.sum(w * proba_unlabeled[indices, col_true]) / w_sum)
            else:
                local_true_support = 1.0

            q_err[j] = confident_wrong * local_uncertainty * local_true_support

        q_err = _safe_minmax_norm(q_err, eps=self.eps)
        return q_err, err_indices

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _normalized_error_propagation(
        self,
        X_unlabeled: np.ndarray,
        X_error_sources: np.ndarray,
        error_strength: np.ndarray,
        k_eff: int,
    ) -> np.ndarray:
        n_u = len(X_unlabeled)
        if n_u == 0 or len(X_error_sources) == 0:
            return np.zeros(n_u, dtype=float)

        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X_unlabeled)

        prop_sum = np.zeros(n_u, dtype=float)
        weight_sum = np.zeros(n_u, dtype=float)

        for e, s in zip(X_error_sources, error_strength):
            s = float(s)
            if s <= 0.0:
                continue

            distances, indices = nbrs.kneighbors(e.reshape(1, -1))
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)

            d_scale = float(np.mean(distances)) + self.eps
            d_norm = distances / d_scale

            alpha_i = self._adaptive_alpha_from_distances(distances)
            w = np.exp(-alpha_i * d_norm)

            prop_sum[indices] += w * s
            weight_sum[indices] += w

        U_prop = prop_sum / (weight_sum + self.eps)
        return _safe_minmax_norm(U_prop, eps=self.eps)

    # ------------------------------------------------------------------
    # Adaptive lambda
    # ------------------------------------------------------------------

    def _effective_lambda(
        self,
        U_base: np.ndarray,
        U_prop: np.ndarray,
        q_err: np.ndarray,
    ) -> float:
        lam = float(np.clip(self.lambda_prop, 0.0, 1.0))

        if (not self.adaptive_lambda) or q_err.size == 0:
            return lam

        n_credible = float(len(q_err))
        phi_count = 1.0 - np.exp(-n_credible / max(self.credible_errors_tau, self.eps))

        if self.use_quality_adaptation:
            phi_quality = float(np.mean(q_err))
        else:
            phi_quality = 1.0

        if self.use_shape_adaptation:
            std_prop = float(np.std(U_prop))
            std_base = float(np.std(U_base)) + self.eps
            phi_shape = float(np.clip(std_prop / std_base, 0.0, 1.0))
        else:
            phi_shape = 1.0

        lam_eff = lam * phi_count * phi_quality * phi_shape
        return float(np.clip(lam_eff, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diversification
    # ------------------------------------------------------------------

    def _select_diverse_from_top(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        n = len(U)
        k = min(int(budget), n)
        if k <= 0:
            return np.array([], dtype=int)

        n_pool = min(n, max(k, int(np.ceil(k * max(1, k_eff) * self.pool_multiplier))))
        top_idx = np.argsort(-U)[:n_pool]
        X_top = X_unlabeled[top_idx]

        if k == 1 or len(X_top) == 1:
            return top_idx[:1].astype(int)

        n_clusters = min(k, len(X_top))
        km = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=self.random_state,
        )

        if self.weighted_kmeans:
            w = np.asarray(U[top_idx], dtype=float)
            if self.weight_power != 1.0:
                w = np.power(w, self.weight_power)
            w = w + self.eps
            km.fit(X_top, sample_weight=w)
        else:
            km.fit(X_top)

        labels = km.labels_

        selected = []
        for c in np.unique(labels):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            best_local = members[np.argmax(U[top_idx[members]])]
            selected.append(int(top_idx[best_local]))

        selected = list(dict.fromkeys(selected))
        if len(selected) < k:
            for idx in top_idx:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= k:
                    break

        return np.asarray(selected[:k], dtype=int)

    # ------------------------------------------------------------------
    # Propagation gate
    # ------------------------------------------------------------------

    def _gate_multiplier(self, U_base: np.ndarray, U_prop: np.ndarray) -> float:
        if not self.use_gate:
            return 1.0

        std_base = float(np.std(U_base)) + self.eps
        std_prop = float(np.std(U_prop))
        shape_ratio = std_prop / std_base

        corr_bp = abs(_safe_corr(U_base, U_prop, eps=self.eps))

        if not self.gate_soft:
            if shape_ratio < self.gate_std_threshold:
                return 0.0
            if corr_bp > self.gate_corr_threshold:
                return 0.0
            return 1.0

        if shape_ratio >= self.gate_std_threshold:
            g_shape = 1.0
        else:
            g_shape = shape_ratio / max(self.gate_std_threshold, self.eps)

        if corr_bp <= self.gate_corr_threshold:
            g_corr = 1.0
        else:
            g_corr = 1.0 - (corr_bp - self.gate_corr_threshold) / max(1.0 - self.gate_corr_threshold, self.eps)
            g_corr = float(np.clip(g_corr, 0.0, 1.0))

        return float(np.clip(g_shape * g_corr, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_l = state.X_labeled
        y_l = state.y_labeled
        X_u = state.X_unlabeled
        model = state.model

        # Stash the live estimator so _source_scores can clone it for
        # the honest (cross-validated) proba estimate.
        self._honest_model_ref = model

        if budget <= 0 or len(X_u) == 0:
            return np.array([], dtype=int)

        k_eff = self._effective_k(len(X_u))
        '''
        # Nombre total de classes globales
        n_classes = int(np.max(state.y_pool)) + 1 if hasattr(state, "y_pool") else None

        # Base uncertainty
        proba_u = _safe_predict_proba(model, X_u, n_classes=n_classes)
        U_base = self._base_uncertainty(proba_u)
        U_base = _safe_minmax_norm(U_base, eps=self.eps)

        if len(X_l) == 0:
            return self._select_diverse_from_top(X_u, U_base, budget, k_eff)

        proba_l = _safe_predict_proba(model, X_l, n_classes=n_classes)

        classes_seen = np.arange(proba_l.shape[1], dtype=int)
        '''
        
        if self.use_safe_proba_mapping:
            n_classes = int(np.max(state.y_pool)) + 1 if hasattr(state, "y_pool") else None
            proba_u = _safe_predict_proba(model, X_u, n_classes=n_classes)
            proba_l = _safe_predict_proba(model, X_l, n_classes=n_classes) if len(X_l) > 0 else np.empty((0, 0))
            classes_seen = np.arange(proba_l.shape[1], dtype=int) if len(X_l) > 0 else None
        else:
            proba_u = model.predict_proba(X_u)
            proba_l = model.predict_proba(X_l) if len(X_l) > 0 else np.empty((0, 0))
            classes_seen = np.asarray(model.classes_, dtype=int) if len(X_l) > 0 else None

        U_base = self._base_uncertainty(proba_u)
        U_base = _safe_minmax_norm(U_base, eps=self.eps)

        q_err, err_indices = self._source_scores(
            X_labeled=X_l,
            y_labeled=y_l,
            proba_labeled=proba_l,
            X_unlabeled=X_u,
            proba_unlabeled=proba_u,
            k_eff=k_eff,
            classes_seen=classes_seen,
        )

        if q_err.size > 0:
            credible_mask = q_err >= float(np.clip(self.min_source_score, 0.0, 1.0))
            q_err = q_err[credible_mask]
            err_indices = err_indices[credible_mask]

        if q_err.size > 0 and self.lambda_prop > 0.0:
            X_error_sources = X_l[err_indices]
            U_prop = self._normalized_error_propagation(
                X_unlabeled=X_u,
                X_error_sources=X_error_sources,
                error_strength=q_err,
                k_eff=k_eff,
            )
        else:
            U_prop = np.zeros(len(X_u), dtype=float)

        lam_eff = self._effective_lambda(U_base=U_base, U_prop=U_prop, q_err=q_err)
        gate_mult = self._gate_multiplier(U_base=U_base, U_prop=U_prop)
        lam_eff *= gate_mult
        lam_eff = float(np.clip(lam_eff, 0.0, 1.0))

        if lam_eff > 0.0:
            U = U_base + lam_eff * U_prop
        else:
            U = U_base.copy()

        return self._select_diverse_from_top(X_u, U, budget, k_eff)