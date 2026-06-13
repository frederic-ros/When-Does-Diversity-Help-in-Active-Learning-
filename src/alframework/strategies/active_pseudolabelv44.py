# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV44 — V4 with adaptive cluster-representative strategy.

Single change vs V4: when picking the representative point of each KMeans
cluster, V4.4 interpolates between two strategies based on the effective
propagation strength (`lam_eff`):

- When `lam_eff` is large  → trust U: pick `argmax(U)` within cluster
                              (V4 behavior, exploits propagation signal)
- When `lam_eff` is small  → trust geometry: pick closest-to-centroid
                              within cluster (dbal behavior, exploits
                              cluster representativeness)

Motivation
----------

After a head-to-head benchmark of V4 vs dbal vs margin across 8 seeds,
4 scenarios, 2 classifiers (16 cells), the failure pattern is clear:

- V4 loses to dbal on RF/medium (Δ = -1.9%, p = 0.004)
- V4 loses to margin on LR/hardhigh (Δ = -1.55%, p = 0.060)

In both losing cases, V4's propagation produces a flat or noisy signal:
RF on a moderately easy task overfits its labeled set, and propagation in
100D is dominated by distance-noise. In these regimes, dbal wins by
picking *representative* points (closest to cluster centroid), which is
what an active learner should do when no good per-point signal is
available.

V4.4 detects this regime via the gated effective lambda already computed
in V4 (`lam_eff = λ × φ_count × φ_quality × φ_shape × gate`). When
`lam_eff < lam_threshold`, V4.4 uses dbal-style selection within
clusters; otherwise it keeps V4's argmax(U) selection.

When `lam_eff = 0` (no propagation), V4.4 reduces to dbal-on-margin.
When `lam_eff > lam_threshold`, V4.4 reduces to V4. The cross-over is
smooth: a soft mixture between argmax(U) and closest-to-centroid is used
in the transition zone.

This is **one new behavior, one new threshold parameter**, and uses
information V4 already computes. No retraining, no new model fitting.

Set `adaptive_representative=False` to recover V4 exactly.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv4 import (
    ActivePseudoLabelV4,
    _safe_minmax_norm,
)
from alframework.core.state import ALState
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


@register("ActivePseudoLabelV44")
class ActivePseudoLabelV44(ActivePseudoLabelV4):
    """
    V4 with adaptive cluster-representative selection.

    Parameters
    ----------
    adaptive_representative : bool, default True
        If True, use lam_eff to interpolate between argmax(U) (V4 style)
        and closest-to-centroid (dbal style) within each KMeans cluster.

    lam_threshold : float, default 0.15
        Below this value of lam_eff, lean toward dbal-style selection.
        Above 2x this value, fully V4-style. Smooth transition between.

    All other parameters inherited from ActivePseudoLabelV4.
    """

    def __init__(
        self,
        *,
        adaptive_representative: bool = True,
        lam_threshold: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adaptive_representative = bool(adaptive_representative)
        self.lam_threshold = float(lam_threshold)
        self._last_lam_eff = 0.0  # cached for diversification step

    # ------------------------------------------------------------------ #
    # Override of public API to capture lam_eff before diversification    #
    # ------------------------------------------------------------------ #

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_l = state.X_labeled
        y_l = state.y_labeled
        X_u = state.X_unlabeled
        model = state.model

        # --- CORRECTIF (bug honest_model_ref) ---
        # V4.4 réimplémente select() sans appeler super().select().
        # La version V4 posait self._honest_model_ref = model pour
        # que _source_scores -> _honest_proba_labeled puisse cloner
        # l'estimateur et calculer la proba CV honnête. Sans cette
        # ligne, _honest_proba_labeled reçoit model=None, sort
        # immédiatement sur la proba MÉMORISÉE, q_err devient vide,
        # lam_eff=0, alpha=1 : la propagation ne s'active JAMAIS.
        self._honest_model_ref = model

        if budget <= 0 or len(X_u) == 0:
            return np.array([], dtype=int)

        k_eff = self._effective_k(len(X_u))

        if self.use_safe_proba_mapping:
            from alframework.strategies.active_pseudolabelv4 import _safe_predict_proba
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

        if len(X_l) == 0:
            self._last_lam_eff = 0.0
            return self._select_diverse_from_top_adaptive(X_u, U_base, budget, k_eff)

        q_err, err_indices = self._source_scores(
            X_labeled=X_l, y_labeled=y_l, proba_labeled=proba_l,
            X_unlabeled=X_u, proba_unlabeled=proba_u,
            k_eff=k_eff, classes_seen=classes_seen,
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

        # Cache for diversification step
        self._last_lam_eff = lam_eff

        if lam_eff > 0.0:
            U = U_base + lam_eff * U_prop
        else:
            U = U_base.copy()

        return self._select_diverse_from_top_adaptive(X_u, U, budget, k_eff)

    # ------------------------------------------------------------------ #
    # Adaptive diversification                                           #
    # ------------------------------------------------------------------ #

    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        Same pool selection and KMeans as V4, but the per-cluster
        representative is selected adaptively:
        - score_for_representative = (1 - alpha) * U_norm  -  alpha * dist_to_centroid_norm
        - alpha = 1 - smoothstep(lam_eff / lam_threshold)
        - alpha=1 → pure dbal (closest to centroid)
        - alpha=0 → pure V4 (argmax U)
        """
        n = len(U)
        k = min(int(budget), n)
        if k <= 0:
            return np.array([], dtype=int)

        # Same pool size logic as V4
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
        centers = km.cluster_centers_

        # Compute alpha (mix between U-argmax and centroid-distance)
        if self.adaptive_representative:
            # Smooth interpolation: alpha=1 when lam_eff << lam_threshold,
            # alpha=0 when lam_eff >= 2 * lam_threshold
            r = self._last_lam_eff / max(self.lam_threshold, self.eps)
            # smoothstep between 0 and 2
            t = float(np.clip(r / 2.0, 0.0, 1.0))
            t_smooth = t * t * (3.0 - 2.0 * t)
            alpha = 1.0 - t_smooth
        else:
            alpha = 0.0

        selected = []
        for c_idx, center in enumerate(centers):
            members = np.where(labels == c_idx)[0]
            if members.size == 0:
                continue

            U_members = U[top_idx[members]]
            X_members = X_top[members]
            d_to_center = np.linalg.norm(X_members - center, axis=1)

            # Normalize within cluster (rank-style for robustness)
            if members.size > 1:
                U_norm = (U_members - U_members.min()) / (U_members.max() - U_members.min() + self.eps)
                d_norm = (d_to_center - d_to_center.min()) / (d_to_center.max() - d_to_center.min() + self.eps)
            else:
                U_norm = np.array([1.0])
                d_norm = np.array([0.0])

            # Score: high U is good (so U_norm), small distance is good (so 1 - d_norm)
            score = (1.0 - alpha) * U_norm + alpha * (1.0 - d_norm)

            best_local = members[int(np.argmax(score))]
            selected.append(int(top_idx[best_local]))

        # Dedup + fill
        selected = list(dict.fromkeys(selected))
        if len(selected) < k:
            for idx in top_idx:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= k:
                    break

        return np.asarray(selected[:k], dtype=int)
