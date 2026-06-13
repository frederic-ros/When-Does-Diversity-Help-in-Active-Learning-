# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV57
====================

V5.7 = WALC + routing structurel par mode, fixé au premier round.

Motivation
----------
Trois familles d'AL par clustering ont des forces complémentaires mais
disjointes :

  DBAL    : KMeans pondéré U + sélection au centroïde (densité)
            → fort sur les problèmes BINAIRES à frontière dense
  Rank2022: Ward + sélection du point le plus incertain
            → fort sur les problèmes MULTICLASSES
  WALC    : KMeans U² + compromis (1−α)·U + α·(1−d)
            → fort quand le classifieur est « plat » (RF-like, mean_U élevé)

Aucune n'est optimale sur tous les régimes :
  - DBAL s'effondre sur le multiclasse (R22 le bat largement)
  - R22 s'effondre sur le binaire dense (DBAL le bat) et a le pire regret
  - WALC seul est moyen sur le LR tabulaire binaire

Principe V5.7
-------------
Plutôt que d'interpoler un paramètre, V5.7 ROUTE vers le mode complet
(clustering + taille de pool + règle de sélection) le mieux adapté au
RÉGIME du problème, détecté à partir de signaux structurels stables au
premier round :

    n_classes ≥ multiclass_thr        → mode R22   (Ward + max-U)
    sinon, mean_U ≥ u_flat_trigger     → mode WALC_K (KMeans U² + compromis)
    sinon (binaire LR-like)            → mode DBAL  (KMeans pondéré + centroïde)

Le mode est FIXÉ au round `route_round` (défaut 1) et conservé pour toute
la run. Les signaux (n_classes, mean_U) sont stables dès les premiers
points, donc le routing est fiable et ne « flotte » pas.

Objectif : performance la plus CONSTANTE sur l'ensemble des régimes
(modèle × structure de classes), c.-à-d. regret maximal borné, plutôt
que l'optimum sur un régime isolé.

Paramètres nouveaux
-------------------
route_round : int = 1
    Round auquel le mode est figé (≥1 pour disposer d'un classifieur
    entraîné sur l'init).

multiclass_thr : int = 5
    n_classes ≥ ce seuil → mode R22.

dbal_pool_mult : int = 5
    Multiplicateur de pool pour le mode DBAL (top k·m points).

r22_pool_mult : int = 10
    Multiplicateur de pool pour le mode R22.

debug_route : bool = False
    Affiche le mode choisi et les signaux au moment du routing.

Héritage
--------
V5.7(ActivePseudoLabelV55) → u_flat_trigger, pm_flat, ward_linkage,
n_init_kmeans, max_selection_pool, etc. tous hérités via **kwargs.
"""

from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv55 import ActivePseudoLabelV55
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


EPS = 1e-12


def _base_uncertainty(proba: np.ndarray) -> np.ndarray:
    """Marge d'incertitude : 1 − (p_max − p_2nd)."""
    s = np.sort(proba, axis=1)
    if s.shape[1] < 2:
        return np.zeros(len(s))
    return 1.0 - (s[:, -1] - s[:, -2])


@register("ActivePseudoLabelV57")
class ActivePseudoLabelV57(ActivePseudoLabelV55):
    """
    V5.7 : routing structurel par mode (R22 / DBAL / WALC_K), fixé round 1.

    Internalise les forces de DBAL (binaire dense), R22 (multiclasse) et
    WALC (RF-like) en choisissant le mode complet selon le régime détecté.
    """

    def __init__(
        self,
        *,
        # ── Paramètres routing V5.7 ──────────────────────────────
        route_round:     int  = 1,
        multiclass_thr:  int  = 5,
        dbal_pool_mult:  int  = 5,
        r22_pool_mult:   int  = 10,
        debug_route:     bool = False,

        # ── Héritage V5.5 / V5.4 / V5.3 ─────────────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.route_round    = max(1, int(route_round))
        self.multiclass_thr = max(2, int(multiclass_thr))
        self.dbal_pool_mult = max(1, int(dbal_pool_mult))
        self.r22_pool_mult  = max(1, int(r22_pool_mult))
        self.debug_route    = bool(debug_route)

        if not hasattr(self, "u_flat_trigger"):
            self.u_flat_trigger = 0.50

        # État interne
        self._v57_mode:        str = None    # mode figé
        self._v57_round_count: int = 0
        self._v57_n_classes:   int = 0

    # ------------------------------------------------------------------
    def _reset_run(self) -> None:
        self._v57_mode = None
        self._v57_round_count = 0
        self._v57_n_classes = 0

    # ------------------------------------------------------------------
    def _decide_mode(self, n_classes: int, mean_u: float) -> str:
        """
        Règle de routing structurel :
          multiclasse        → R22
          RF-like (U plat)   → WALC_K
          binaire LR-like    → DBAL
        """
        if n_classes >= self.multiclass_thr:
            return "R22"
        if mean_u >= self.u_flat_trigger:
            return "WALC_K"
        return "DBAL"

    # ------------------------------------------------------------------
    def select(self, state, budget: int) -> np.ndarray:
        X_lab = state.X_labeled
        y_lab = state.y_labeled
        clf   = state.model

        # Reset au début d'un run
        if self._v57_round_count == 0:
            self._reset_run()
        self._v57_round_count += 1

        n_classes = len(np.unique(y_lab))
        self._v57_n_classes = n_classes

        # Décider le mode au round `route_round`, puis le figer
        if self._v57_mode is None and self._v57_round_count >= self.route_round:
            try:
                proba = clf.predict_proba(state.X_unlabeled)
                U = _safe_minmax_norm(_base_uncertainty(proba), eps=EPS)
                mean_u = float(np.mean(U))
            except Exception:
                mean_u = 0.0
            self._v57_mode = self._decide_mode(n_classes, mean_u)
            if self.debug_route:
                print(
                    f"[V57-route] round={self._v57_round_count}  "
                    f"n_cls={n_classes}  mean_U={mean_u:.3f}  "
                    f"→ mode={self._v57_mode}"
                )

        # Référence modèle pour recalcul U margin pur dans modes R22/DBAL
        self._v57_model_ref = clf

        return super().select(state, budget)

    # ------------------------------------------------------------------
    def _pure_margin_U(self, X_pts: np.ndarray) -> np.ndarray:
        """
        Incertitude margin PURE (1 − (p_max − p_2nd)), sans mélange
        d'entropie ni normalisation min-max, pour reproduire fidèlement
        DBAL et R22 qui utilisent le margin brut. Renvoie None si indispo.
        """
        clf = getattr(self, "_v57_model_ref", None)
        if clf is None:
            return None
        try:
            return _base_uncertainty(clf.predict_proba(X_pts))
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        Sélection selon le mode figé. Si mode non encore décidé,
        comportement V5.5 par défaut.
        """
        U = np.asarray(U, dtype=float)
        n = len(U)
        k = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        mode = self._v57_mode

        # Avant routing : comportement V5.5 standard
        if mode is None:
            return super()._select_diverse_from_top_adaptive(
                X_unlabeled, U, budget, k_eff
            )

        # ── Mode R22 : Ward + max(U) ──────────────────────────────
        if mode == "R22":
            # Margin pur (comme R22 référence), fallback sur U hérité
            U_pure = self._pure_margin_U(X_unlabeled)
            U_r = U_pure if U_pure is not None and len(U_pure) == n else U
            n_top = min(n, max(k, k * self.r22_pool_mult))
            top   = np.argsort(-U_r)[:n_top]
            X_top = X_unlabeled[top]
            nc    = min(k, len(X_top))
            if nc <= 1:
                return top[:k]
            try:
                from sklearn.cluster import AgglomerativeClustering
                labels = AgglomerativeClustering(
                    n_clusters=nc, linkage=self.ward_linkage
                ).fit_predict(X_top)
                sel = []
                for c in range(nc):
                    m = np.where(labels == c)[0]
                    if not len(m):
                        continue
                    sel.append(int(top[m[int(np.argmax(U_r[top[m]]))]]))
                return np.array(list(dict.fromkeys(sel))[:k], dtype=int)
            except Exception:
                return top[:k]

        # ── Mode DBAL : KMeans pondéré U + centroïde ──────────────
        # DBAL conserve l'incertitude héritée (margin+entropie mixés de
        # WALC) : sur les binaires, ce signal mixte améliore la sélection
        # au centroïde (gain net mesuré, ex. credit +2.3pp) sans nuire
        # ailleurs. Seul R22 exige le margin pur (cf. multiclasse).
        if mode == "DBAL":
            U_d = U
            n_top = min(n, max(k, k * self.dbal_pool_mult))
            top   = np.argsort(-U_d)[:n_top]
            X_top = X_unlabeled[top]
            nc    = min(k, len(X_top))
            if nc <= 1:
                return top[:k]
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=nc,
                n_init=self.n_init_kmeans,
                random_state=self.random_state,
            ).fit(X_top, sample_weight=U_d[top] + EPS)
            sel = []
            for ctr in km.cluster_centers_:
                d = np.linalg.norm(X_top - ctr, axis=1)
                sel.append(int(top[int(np.argmin(d))]))
            return np.array(list(dict.fromkeys(sel))[:k], dtype=int)

        # ── Mode WALC_K : KMeans U² + compromis (V5.5 KMeans) ─────
        # On force le chemin KMeans de V5.5 en marquant le classifieur
        # comme "plat" pour cette sélection.
        n_top = min(n, max(k, k * self.pm_flat))
        top   = np.argsort(-U)[:n_top]
        X_top = X_unlabeled[top]
        nc    = min(k, len(X_top))
        if nc <= 1:
            return top[:k]
        cv    = float(np.std(U) / (np.mean(U) + EPS))
        alpha = float(np.clip(0.6 - cv * 0.5, 0.1, 0.6))
        from sklearn.cluster import KMeans
        w  = np.power(U[top] + EPS, 2.0)
        km = KMeans(
            n_clusters=nc,
            n_init=self.n_init_kmeans,
            random_state=self.random_state,
        ).fit(X_top, sample_weight=w)
        sel = []
        for c, ctr in enumerate(km.cluster_centers_):
            m = np.where(km.labels_ == c)[0]
            if not len(m):
                continue
            Un = _safe_minmax_norm(U[top[m]], eps=EPS)
            dn = _safe_minmax_norm(
                np.linalg.norm(X_top[m] - ctr, axis=1), eps=EPS
            )
            sel.append(int(top[m[int(np.argmax(
                (1 - alpha) * Un + alpha * (1 - dn)
            ))]]))
        return np.array(list(dict.fromkeys(sel))[:k], dtype=int)

    # ------------------------------------------------------------------
    @property
    def routed_mode(self) -> str:
        """Mode figé pour la run courante (R22 / DBAL / WALC_K)."""
        return self._v57_mode

    @property
    def route_info(self) -> dict:
        return {
            "mode":           self._v57_mode,
            "n_classes":      self._v57_n_classes,
            "multiclass_thr": self.multiclass_thr,
            "u_flat_trigger": self.u_flat_trigger,
        }
