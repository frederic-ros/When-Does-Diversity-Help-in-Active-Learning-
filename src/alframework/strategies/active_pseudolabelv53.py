# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV53
====================

V53 = V5.2 avec accélération de la propagation, sans changer le principe.

Objectif
--------
Garder la logique scientifique V5.2 :
- backbone V5.1 / V4.4
- routeur alpha V4.4 <-> V5.1
- propagation d'erreurs quand elle est utile

Mais éviter de payer systématiquement le coût complet :
- source scoring honnête / voisinage
- propagation vers tout X_unlabeled

Accélérations ajoutées
----------------------
1. fast_prop_guard : pré-gate sur U_base.
   Si le signal d'incertitude est déjà très contrasté, on saute toute la
   propagation : pas de _source_scores, pas de _normalized_error_propagation.

2. skip_source_scores_when_no_prop : si lambda_prop <= 0, skip réel.
   Cela permet un vrai mode no-prop rapide.

3. max_prop_sources : cap sur les sources d'erreur propagées.
   On garde les sources q_err les plus crédibles.

4. prop_on_top_pool : propagation seulement sur le top-pool candidat.
   Au lieu de diffuser vers tout X_u, on diffuse vers les points qui ont une
   chance réaliste d'être sélectionnés.

Ces changements ne modifient pas la sélection finale. Ils changent seulement
quand et où on paie le coût de propagation.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.active_pseudolabelv51 import ActivePseudoLabelV51
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


@register("ActivePseudoLabelV53")
class ActivePseudoLabelV53(ActivePseudoLabelV51):
    """V53: V5.2 router + fast propagation guards."""

    def __init__(
        self,
        *,
        # V5.2 router
        router_enabled: bool = True,
        v44_lam_min: float = 0.12,
        v44_contrast_min: float = 0.30,
        highdim_v44_min: float = 0.65,
        highdim_flatness_max: float = 0.85,
        debug_router: bool = False,

        # Panel aliases / convenience names
        contrast_threshold: float | None = None,
        flatness_threshold: float | None = None,
        highdim_threshold: float | None = None,
        n_init_kmeans: int | str = "auto",
        adaptive_weight_power: bool = True,
        peakedness_factor: float = 1.2,

        # Speed controls
        source_policy: str = "normal",
        #   "normal"    : comportement V5.3 standard avec fast_prop_guard
        #   "auto_fast" : saute _source_scores si U_base n'est pas ambigu
        #   "never"     : saute toujours _source_scores / propagation
        #   "always"    : force le chemin source_scores sauf lambda_prop=0
        max_u_contrast_for_source: float = 0.28,
        min_u_flatness_for_source: float = 0.88,
        max_selection_pool: int | None = None,

        skip_source_scores_when_no_prop: bool = True,
        fast_prop_guard: bool = True,
        min_u_contrast_skip: float = 0.35,
        min_u_flatness_skip: float | None = None,
        min_qerr_mean_for_prop: float = 0.03,
        min_qerr_max_for_prop: float = 0.08,
        max_prop_sources: int | None = 32,
        prop_on_top_pool: bool = True,
        prop_pool_multiplier: float = 8.0,
        min_prop_pool: int = 64,
        debug_fast: bool = False,
        **kwargs,
    ):
        if contrast_threshold is not None:
            v44_contrast_min = float(contrast_threshold)
        if highdim_threshold is not None:
            highdim_v44_min = float(highdim_threshold)
        if flatness_threshold is not None:
            highdim_flatness_max = float(max(highdim_flatness_max, flatness_threshold))

        super().__init__(**kwargs)

        # Router
        self.router_enabled = bool(router_enabled)
        self.v44_lam_min = float(v44_lam_min)
        self.v44_contrast_min = float(v44_contrast_min)
        self.highdim_v44_min = float(highdim_v44_min)
        self.highdim_flatness_max = float(highdim_flatness_max)
        self.debug_router = bool(debug_router)

        # KMeans
        self.n_init_kmeans = n_init_kmeans
        self.adaptive_weight_power = bool(adaptive_weight_power)
        self.peakedness_factor = float(peakedness_factor)

        # Fast propagation
        self.source_policy = str(source_policy).lower()
        if self.source_policy not in {"normal", "auto_fast", "never", "always"}:
            raise ValueError(
                "source_policy must be one of: 'normal', 'auto_fast', 'never', 'always'"
            )
        self.max_u_contrast_for_source = float(max_u_contrast_for_source)
        self.min_u_flatness_for_source = float(min_u_flatness_for_source)
        self.max_selection_pool = None if max_selection_pool is None else int(max_selection_pool)

        self.skip_source_scores_when_no_prop = bool(skip_source_scores_when_no_prop)
        self.fast_prop_guard = bool(fast_prop_guard)
        self.min_u_contrast_skip = float(min_u_contrast_skip)
        self.min_u_flatness_skip = None if min_u_flatness_skip is None else float(min_u_flatness_skip)
        self.min_qerr_mean_for_prop = float(min_qerr_mean_for_prop)
        self.min_qerr_max_for_prop = float(min_qerr_max_for_prop)
        self.max_prop_sources = None if max_prop_sources is None else int(max_prop_sources)
        self.prop_on_top_pool = bool(prop_on_top_pool)
        self.prop_pool_multiplier = float(prop_pool_multiplier)
        self.min_prop_pool = int(min_prop_pool)
        self.debug_fast = bool(debug_fast)

        # Diagnostics
        self._v52_mode = "V51"
        self._v52_router_reason = "default"
        self._v53_prop_mode = "unset"
        self._v53_prop_pool_size = 0
        self._v53_source_count = 0
        self._v53_qerr_mean = 0.0
        self._v53_qerr_max = 0.0
        self._v53_u_contrast = 0.0
        self._v53_u_flatness = 0.0

    # ------------------------------------------------------------------
    # Public API with fast propagation path
    # ------------------------------------------------------------------
    def select(self, state: ALState, budget: int) -> np.ndarray:
        self._v51_y_labeled = np.asarray(state.y_labeled)

        X_l = state.X_labeled
        y_l = state.y_labeled
        X_u = state.X_unlabeled
        model = state.model

        # Needed by V4 honest source scoring when enabled.
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
            self._v53_prop_mode = "no_labeled"
            return self._select_diverse_from_top_adaptive(X_u, U_base, budget, k_eff)

        # Cheap diagnostics before the expensive source/propagation path.
        self._v53_u_contrast = self._score_contrast(U_base)
        self._v53_u_flatness = self._score_flatness(U_base)

        skip_all_prop = False
        skip_reason = ""

        if self.source_policy == "never":
            skip_all_prop = True
            skip_reason = "source_policy_never"

        elif self.skip_source_scores_when_no_prop and float(self.lambda_prop) <= 0.0:
            skip_all_prop = True
            skip_reason = "lambda_prop_zero"

        elif self.source_policy == "auto_fast":
            # Run source scoring only when the uncertainty landscape is ambiguous:
            # low contrast AND high flatness. Otherwise skip the expensive path.
            if (
                self._v53_u_contrast > self.max_u_contrast_for_source
                or self._v53_u_flatness < self.min_u_flatness_for_source
            ):
                skip_all_prop = True
                skip_reason = "auto_fast_clear_uncertainty"

        elif self.fast_prop_guard and self.source_policy != "always":
            # If uncertainty already has a clear ranking, V4/V5.2 can use it
            # directly; propagation is often costly and redundant.
            if self._v53_u_contrast >= self.min_u_contrast_skip:
                skip_all_prop = True
                skip_reason = "u_contrast_high"

            # Optional extra guard: if U is explicitly not flat, skip.
            if (
                not skip_all_prop
                and self.min_u_flatness_skip is not None
                and self._v53_u_flatness <= self.min_u_flatness_skip
            ):
                skip_all_prop = True
                skip_reason = "u_not_flat"

        if skip_all_prop:
            q_err = np.array([], dtype=float)
            err_indices = np.array([], dtype=int)
            U_prop = np.zeros(len(X_u), dtype=float)
            lam_eff = 0.0
            self._v53_prop_mode = f"skipped:{skip_reason}"
            self._v53_source_count = 0
            self._v53_qerr_mean = 0.0
            self._v53_qerr_max = 0.0
        else:
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

            self._v53_source_count = int(q_err.size)
            self._v53_qerr_mean = float(np.mean(q_err)) if q_err.size else 0.0
            self._v53_qerr_max = float(np.max(q_err)) if q_err.size else 0.0

            # Cheap gate after source scoring but before propagation.
            weak_sources = (
                q_err.size == 0
                or float(self.lambda_prop) <= 0.0
                or self._v53_qerr_mean < self.min_qerr_mean_for_prop
                or self._v53_qerr_max < self.min_qerr_max_for_prop
            )

            if weak_sources:
                U_prop = np.zeros(len(X_u), dtype=float)
                lam_eff = 0.0
                self._v53_prop_mode = "skipped:weak_sources"
            else:
                # Cap sources to reduce propagation cost and source noise.
                if self.max_prop_sources is not None and q_err.size > self.max_prop_sources:
                    keep = np.argsort(-q_err)[: self.max_prop_sources]
                    q_err = q_err[keep]
                    err_indices = err_indices[keep]

                X_error_sources = X_l[err_indices]

                if self.prop_on_top_pool:
                    # Propagate only to candidates that can realistically be selected.
                    n_prop_pool = min(
                        len(X_u),
                        max(
                            int(self.min_prop_pool),
                            int(np.ceil(max(1, int(budget)) * max(1, k_eff) * self.prop_pool_multiplier)),
                        ),
                    )
                    top_prop_idx = np.argsort(-U_base)[:n_prop_pool]
                    U_prop_top = self._normalized_error_propagation(
                        X_unlabeled=X_u[top_prop_idx],
                        X_error_sources=X_error_sources,
                        error_strength=q_err,
                        k_eff=k_eff,
                    )
                    U_prop = np.zeros(len(X_u), dtype=float)
                    U_prop[top_prop_idx] = U_prop_top
                    self._v53_prop_pool_size = int(n_prop_pool)
                    self._v53_prop_mode = "top_pool"
                else:
                    U_prop = self._normalized_error_propagation(
                        X_unlabeled=X_u,
                        X_error_sources=X_error_sources,
                        error_strength=q_err,
                        k_eff=k_eff,
                    )
                    self._v53_prop_pool_size = int(len(X_u))
                    self._v53_prop_mode = "full"

                lam_eff = self._effective_lambda(U_base=U_base, U_prop=U_prop, q_err=q_err)
                gate_mult = self._gate_multiplier(U_base=U_base, U_prop=U_prop)
                lam_eff *= gate_mult
                lam_eff = float(np.clip(lam_eff, 0.0, 1.0))

        self._last_lam_eff = float(lam_eff)

        if lam_eff > 0.0:
            U = U_base + lam_eff * U_prop
        else:
            U = U_base.copy()

        if self.debug_fast:
            print(
                "V53",
                "prop_mode=", self._v53_prop_mode,
                "u_contrast=", round(self._v53_u_contrast, 4),
                "u_flatness=", round(self._v53_u_flatness, 4),
                "sources=", self._v53_source_count,
                "q_mean=", round(self._v53_qerr_mean, 4),
                "q_max=", round(self._v53_qerr_max, 4),
                "lam_eff=", round(float(lam_eff), 4),
                "pool=", self._v53_prop_pool_size,
            )

        return self._select_diverse_from_top_adaptive(X_u, U, budget, k_eff)

    # ------------------------------------------------------------------
    # Final selection: same V5.2/V5.1 style, configurable KMeans init.
    # ------------------------------------------------------------------
    def _compute_adaptive_power(self, U: np.ndarray) -> float:
        """
        Calcule le weight_power optimal selon la peakedness de U.

        Sur RF : U bimodale (piquée) → q90/q50 élevé → p faible (0.5-1.0)
                 → adoucit la pondération U^p pour éviter la sur-concentration
        Sur LR : U lisse → q90/q50 plus bas → p proche de 2.0
                 → comportement original V5.3

        peakedness = q90 / q50  (ratio des quantiles)
        p = 2.0 / (1 + peakedness_factor × max(0, peakedness - 1))
        clippé dans [0.5, 2.0]
        """
        q90 = float(np.percentile(U, 90))
        q50 = float(np.percentile(U, 50)) + self.eps
        peakedness = q90 / q50
        p = 2.0 / (1.0 + self.peakedness_factor * max(0.0, peakedness - 1.0))
        return float(np.clip(p, 0.5, 2.0))

    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        n = len(U)
        k = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        n_pool = min(
            n,
            max(k, int(np.ceil(k * max(1, k_eff) * self.pool_multiplier))),
        )
        if self.max_selection_pool is not None:
            n_pool = min(n_pool, max(k, int(self.max_selection_pool)))
        top_idx = np.argsort(-U)[:n_pool]
        X_top = X_unlabeled[top_idx]

        n_clusters = min(k, len(X_top))
        if n_clusters <= 1:
            return top_idx[:k].astype(int)

        km = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init_kmeans,
            random_state=self.random_state,
        )

        if self.weighted_kmeans:
            w = np.asarray(U[top_idx], dtype=float)
            # Power adaptatif ou fixe
            p = (self._compute_adaptive_power(U)
                 if self.adaptive_weight_power
                 else self.weight_power)
            if p != 1.0:
                w = np.power(w, p)
            w = w + self.eps
            km.fit(X_top, sample_weight=w)
        else:
            km.fit(X_top)

        labels = km.labels_
        centers = km.cluster_centers_

        alpha = self._effective_alpha_v51(X_unlabeled=X_unlabeled, U=U)

        selected: list[int] = []
        for c_idx, center in enumerate(centers):
            members = np.where(labels == c_idx)[0]
            if members.size == 0:
                continue

            X_m = X_top[members]
            U_m = U[top_idx[members]]
            d_m = np.linalg.norm(X_m - center, axis=1)

            U_norm = _safe_minmax_norm(U_m, eps=self.eps)
            d_norm = _safe_minmax_norm(d_m, eps=self.eps)
            repr_norm = 1.0 - d_norm

            score = (1.0 - alpha) * U_norm + alpha * repr_norm
            best_local = members[int(np.argmax(score))]
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
    # V5.2 alpha router.
    # ------------------------------------------------------------------
    def _effective_alpha_v51(
        self,
        *,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
    ) -> float:
        alpha_base = self._alpha_from_lam_eff()
        self._v51_alpha_base = float(alpha_base)

        if not self.router_enabled:
            self._v52_mode = "V51"
            self._v52_router_reason = "router_disabled"
            return super()._effective_alpha_v51(X_unlabeled=X_unlabeled, U=U)

        flatness = self._score_flatness(U)
        contrast = self._score_contrast(U)
        highdim = self._highdim_pressure(X_unlabeled)
        imbalance = self._labeled_imbalance_from_cache()
        lam_eff = float(getattr(self, "_last_lam_eff", 0.0))

        self._v51_flatness = flatness
        self._v51_contrast = contrast
        self._v51_highdim = highdim
        self._v51_imbalance = imbalance

        if lam_eff >= self.v44_lam_min and contrast >= self.v44_contrast_min:
            alpha = float(np.clip(alpha_base, self.alpha_min, self.alpha_max))
            self._v51_alpha_final = alpha
            self._v52_mode = "V44"
            self._v52_router_reason = "strong_lam_and_contrast"
            return alpha

        if highdim >= self.highdim_v44_min and flatness <= self.highdim_flatness_max:
            alpha = float(np.clip(alpha_base, self.alpha_min, self.alpha_max))
            self._v51_alpha_final = alpha
            self._v52_mode = "V44"
            self._v52_router_reason = "highdim_guard"
            return alpha

        self._v52_mode = "V51"
        self._v52_router_reason = "weak_or_flat_signal"
        return super()._effective_alpha_v51(X_unlabeled=X_unlabeled, U=U)
