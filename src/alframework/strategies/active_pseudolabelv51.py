# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV51
====================

V51 = V4.4 comme cœur + micro-adaptation V5, sans double-diversification.

Objectif
--------
Conserver la stabilité empirique de V4.4 tout en récupérant une partie de la
robustesse de V5 dans les régimes où le signal actif est faible, plat, bruité,
déséquilibré ou en haute dimension.

Différence avec V5
------------------
V5 mélangeait plusieurs forces en même temps :
- score signal U
- représentativité centroïde
- densité locale
- contrôleur de diversité additif
- sélection intra-cluster adaptative

Cela peut créer une double-diversification : le score externe pousse déjà vers
le centroïde, puis l'alpha intra-cluster repousse encore vers le centroïde.
Sur les scénarios faciles / redondants / bien structurés, cela peut diluer le
signal U.

V51 fait plus simple :
- backbone V4.4
- même logique de sélection intra-cluster : U vs centroïde
- alpha est légèrement ajusté par quelques diagnostics V5 :
  flatness, contrast, imbalance, highdim
- pas de density score
- pas de diversity_weight externe
- pas de deuxième couche de scoring

Règle intuitive
---------------
alpha = poids de représentativité centroïde.

- alpha proche de 0 : sélection guidée par U, style V4
- alpha proche de 1 : sélection guidée par centroïde, style DBAL

V51 part du alpha V4.4 basé sur lam_eff, puis applique de petits correctifs :
- si U est plat       -> alpha augmente
- si U est contrasté  -> alpha diminue
- si labels déséquilibrés -> alpha augmente un peu
- si haute dimension  -> alpha diminue un peu, car les centroïdes sont moins fiables

Pourquoi cette version
----------------------
Les résultats fournis suggèrent :
- V4.4 est le meilleur core stable
- V5 aide dans les cas noisy / sparse / rare / multiclass
- mais V5 semble parfois trop diversifier

V51 essaie donc de garder le meilleur de V4.4 et seulement les garde-fous utiles
issus de V5.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.active_pseudolabelv44 import ActivePseudoLabelV44
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


@register("ActivePseudoLabelV51")
class ActivePseudoLabelV51(ActivePseudoLabelV44):
    """
    V51 : V4.4 + alpha controller léger inspiré de V5.

    Paramètres principaux
    ---------------------
    adaptive_representative : bool, default True
        Active le mélange U vs centroïde hérité de V4.4.

    lam_threshold : float, default 0.15
        Seuil V4.4 pour convertir lam_eff en alpha.

    adaptive_alpha : bool, default True
        Active les petits correctifs V51 sur alpha.

    flatness_alpha_weight : float, default 0.15
        Augmente alpha quand le score U est plat.

    contrast_alpha_weight : float, default 0.10
        Diminue alpha quand le score U est très contrasté.

    imbalance_alpha_weight : float, default 0.08
        Augmente alpha si le set labellisé est déséquilibré.

    highdim_alpha_weight : float, default 0.10
        Diminue alpha en haute dimension, car les centroïdes sont moins fiables.

    alpha_min / alpha_max : float
        Bornes finales de alpha.

    Notes
    -----
    Cette classe réutilise select() de V4.4. Elle surcharge uniquement la
    diversification finale afin de remplacer alpha V4.4 par alpha V51.
    """

    def __init__(
        self,
        *,
        adaptive_representative: bool = True,
        lam_threshold: float = 0.15,
        adaptive_alpha: bool = True,
        flatness_alpha_weight: float = 0.15,
        contrast_alpha_weight: float = 0.10,
        imbalance_alpha_weight: float = 0.08,
        highdim_alpha_weight: float = 0.10,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        highdim_start: int = 40,
        highdim_full: int = 200,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(
            adaptive_representative=adaptive_representative,
            lam_threshold=lam_threshold,
            random_state=random_state,
            **kwargs,
        )

        self.adaptive_alpha = bool(adaptive_alpha)
        self.flatness_alpha_weight = float(flatness_alpha_weight)
        self.contrast_alpha_weight = float(contrast_alpha_weight)
        self.imbalance_alpha_weight = float(imbalance_alpha_weight)
        self.highdim_alpha_weight = float(highdim_alpha_weight)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.highdim_start = int(highdim_start)
        self.highdim_full = int(highdim_full)

        # Diagnostics utiles pour debug / benchmark
        self._v51_alpha_base = 0.0
        self._v51_alpha_final = 0.0
        self._v51_flatness = 0.0
        self._v51_contrast = 0.0
        self._v51_imbalance = 0.0
        self._v51_highdim = 0.0

    # ------------------------------------------------------------------
    # Sélection finale : V4.4 core, alpha V51
    # ------------------------------------------------------------------

    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        Même structure que V4.4, mais alpha est ajusté par des diagnostics
        simples issus de V5.

        Score intra-cluster :
            score = (1 - alpha) * U_norm + alpha * repr_norm

        où :
            repr_norm = 1 - distance_to_centroid_norm
        """
        U = np.asarray(U, dtype=float)
        n = len(U)
        k = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        # Même pool size logic que V4.4
        n_pool = min(
            n,
            max(k, int(np.ceil(k * max(1, k_eff) * self.pool_multiplier))),
        )
        top_idx = np.argsort(-U)[:n_pool]
        X_top = X_unlabeled[top_idx]

        n_clusters = min(k, len(X_top))
        if n_clusters <= 1:
            return top_idx[:k].astype(int)

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

        alpha = self._effective_alpha_v51(
            X_unlabeled=X_unlabeled,
            U=U,
        )

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

        # Dedup + fill comme V4.4
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
    # Alpha controller V51
    # ------------------------------------------------------------------

    def _effective_alpha_v51(
        self,
        *,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
    ) -> float:
        """
        Alpha final = alpha V4.4 + petits correctifs V51.

        Important : les poids par défaut sont volontairement modestes.
        L'objectif est de corriger V4.4 dans les régimes difficiles, pas de
        transformer le comportement global en V5.
        """
        alpha = self._alpha_from_lam_eff()
        self._v51_alpha_base = float(alpha)

        if not self.adaptive_alpha:
            alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
            self._v51_alpha_final = alpha
            return alpha

        flatness = self._score_flatness(U)
        contrast = self._score_contrast(U)
        highdim = self._highdim_pressure(X_unlabeled)
        imbalance = self._labeled_imbalance_from_cache()

        self._v51_flatness = flatness
        self._v51_contrast = contrast
        self._v51_highdim = highdim
        self._v51_imbalance = imbalance

        # Règle agile :
        # - U plat       => on fait plus confiance au centroïde
        # - U contrasté  => on fait plus confiance au signal U
        # - imbalance    => un peu plus de représentativité
        # - highdim      => moins de centroïde, car distances moins fiables
        alpha += self.flatness_alpha_weight * flatness
        alpha -= self.contrast_alpha_weight * contrast
        alpha += self.imbalance_alpha_weight * imbalance
        alpha -= self.highdim_alpha_weight * highdim

        alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
        self._v51_alpha_final = alpha
        return alpha

    def _alpha_from_lam_eff(self) -> float:
        """
        Reprise exacte de l'esprit V4.4 :
        - lam_eff faible  -> alpha élevé, DBAL-like
        - lam_eff fort    -> alpha faible, V4-like
        Transition smoothstep entre 0 et 2 * lam_threshold.
        """
        if not self.adaptive_representative:
            return 0.0

        lam_eff = float(getattr(self, "_last_lam_eff", 0.0))
        r = lam_eff / max(self.lam_threshold, self.eps)
        t = float(np.clip(r / 2.0, 0.0, 1.0))
        t_smooth = t * t * (3.0 - 2.0 * t)
        return float(1.0 - t_smooth)

    # ------------------------------------------------------------------
    # Diagnostics V5 simplifiés
    # ------------------------------------------------------------------

    def _score_flatness(self, scores: np.ndarray) -> float:
        """0 = signal concentré / 1 = signal très plat."""
        s = np.asarray(scores, dtype=float)
        if s.size <= 1:
            return 0.0

        s = s - np.min(s)
        total = float(np.sum(s))

        if total <= self.eps:
            return 1.0

        p = np.clip(s / total, self.eps, 1.0)
        ent = -float(np.sum(p * np.log(p)))
        ent_max = float(np.log(len(p)))

        if ent_max <= self.eps:
            return 0.0

        return float(np.clip(ent / ent_max, 0.0, 1.0))

    def _score_contrast(self, scores: np.ndarray) -> float:
        """0 = peu de contraste / 1 = signal très contrasté."""
        s = np.asarray(scores, dtype=float)
        if s.size <= 1:
            return 0.0

        mean = float(np.mean(s))
        if mean <= self.eps:
            return 0.0

        cv = float(np.std(s)) / (mean + self.eps)
        return float(np.clip(cv, 0.0, 1.0))

    def _highdim_pressure(self, X_unlabeled: np.ndarray) -> float:
        """
        0 = dimension faible / 1 = haute dimension.

        Par défaut :
        - 40 features  -> début de garde-fou
        - 200 features -> garde-fou plein
        """
        X = np.asarray(X_unlabeled)
        n_features = X.shape[1] if X.ndim == 2 else 1

        denom = max(1, self.highdim_full - self.highdim_start)
        return float(np.clip((n_features - self.highdim_start) / denom, 0.0, 1.0))

    def _labeled_imbalance_from_cache(self) -> float:
        """
        Essaie de récupérer y_labeled depuis l'état courant.

        Comme V4.4 ne cache pas y_labeled, V51 surcharge select() pour poser
        self._v51_y_labeled avant d'appeler le select() parent.
        """
        y = getattr(self, "_v51_y_labeled", None)
        if y is None:
            return 0.0
        return self._imbalance_proxy(np.asarray(y))

    def _imbalance_proxy(self, y_labeled: np.ndarray) -> float:
        """0 = équilibré / 1 = très déséquilibré."""
        y = np.asarray(y_labeled)
        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        if counts.size <= 1:
            return 1.0

        return float(np.clip(1.0 - counts.min() / counts.max(), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API override léger : cache y_labeled pour imbalance
    # ------------------------------------------------------------------

    def select(self, state: ALState, budget: int) -> np.ndarray:
        self._v51_y_labeled = np.asarray(state.y_labeled)
        return super().select(state, budget)
