# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV55
====================

V5.5 = V5.4 + routing adaptatif du mécanisme de clustering.

Nouveauté par rapport à V5.4
-----------------------------
V5.4 routait déjà le mécanisme *smart* (repondération U + garde) selon
le comportement du classifieur (mean(U) ≥ u_flat_trigger → classifieur
"plat", mean(U) < u_flat_trigger → classifieur "discriminant").

V5.5 utilise le **même signal u_flat_trigger** pour router également le
**mécanisme de clustering** :

  Classifieur plat (RF-like, mean(U) ≥ u_flat_trigger)
    → KMeans pondéré U²  +  filtre pool×pm_flat (défaut 8)
    → optimal quand U est informatif et concentré sur la frontière

  Classifieur discriminant (LR-like, mean(U) < u_flat_trigger)
    → AgglomerativeClustering Ward  +  filtre pool×pm_disc (défaut 20)
    → Ward crée des partitions géométriques uniformes dans un pool
      élargi, offrant une meilleure couverture spatiale quand U est
      déjà bien structuré

Pourquoi ces deux configs ?
  - Sur RF (U bimodal, élevé partout), le KMeans U² exploite le signal
    incertitude fort. Un pool large diluerait ce signal.
  - Sur LR (U lisse, distribué), Ward sur pool élargi couvre mieux
    l'espace d'entrée. L'incertitude seule ne suffit pas à différencier
    les candidats.

Paramètres nouveaux
-------------------
ward_enabled : bool = True
    Active/désactive le routing Ward (sinon V5.4 pur pour tous).

pm_flat : int = 8
    Pool multiplier pour le régime "plat" (KMeans U²).

pm_disc : int = 10
    Pool multiplier pour le régime "discriminant" (Ward).
    Valeur 10 (identique à Rank2022) : assez large pour diversifier
    géométriquement sans dépasser le pool sur les petits datasets
    (n~300-500, pool~200-350 : top-N = min(pool, k×10) = 200 pts).

ward_linkage : str = "ward"
    Linkage utilisé pour AgglomerativeClustering.

debug_clustering : bool = False
    Log du clustering choisi à chaque round.

Héritage
--------
V5.5(ActivePseudoLabelV54) → tous les paramètres V5.4 + V5.3 passent
en **kwargs. Le routing V5.4 (smart + garde) est conservé.

Note : le u_flat_trigger est partagé entre les deux routings (smart V5.4
et clustering V5.5) — même seuil, même signal, cohérence garantie.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
    _HAS_AGGLO = True
except ImportError:
    _HAS_AGGLO = False

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv54 import ActivePseudoLabelV54
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


EPS = 1e-12


@register("ActivePseudoLabelV55")
class ActivePseudoLabelV55(ActivePseudoLabelV54):
    """
    V5.5 : V5.4 + routing adaptatif du clustering (KMeans U² vs Ward).
    Même u_flat_trigger pour le smart (V5.4) et pour le clustering (V5.5).
    """

    def __init__(
        self,
        *,
        # ── Mécanisme clustering adaptatif V5.5 ─────────────────
        ward_enabled: bool = True,
        pm_flat: int = 8,
        pm_disc: int = 10,
        ward_linkage: str = "ward",
        debug_clustering: bool = False,

        # ── Héritage V5.4 + V5.3 (pass-through complet) ─────────
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ward_enabled     = bool(ward_enabled)
        self.pm_flat          = int(pm_flat)
        self.pm_disc          = int(pm_disc)
        self.ward_linkage     = str(ward_linkage)
        self.debug_clustering = bool(debug_clustering)

        # Diagnostics V5.5
        self._v55_clustering_used : str  = "kmeans"   # "kmeans" ou "ward"
        self._v55_pm_used         : int  = pm_flat

    # ------------------------------------------------------------------
    # Override : _select_diverse_from_top_adaptive
    # On surcharge la méthode V5.3 (grand-parent) plutôt que V5.4 car
    # V5.4 appelle super()._select_diverse_from_top_adaptive en fin de
    # sa propre méthode. En héritant de V5.4, notre override est appelé
    # à la place de la méthode V5.3 → le routing clustering s'intercale
    # *avant* la sélection finale, pendant que V5.4 gère smart+guard.
    #
    # Séquence effective :
    #   V5.4._select_diverse_from_top_adaptive(U_adj, ...)
    #     → appelle super() = V5.5._select_diverse_from_top_adaptive
    #       → routing clustering → batch
    #     → V5.4 applique guard post-hoc
    # ------------------------------------------------------------------
    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        Routing clustering selon u_flat :
          - Classifieur plat  → KMeans U² (pm_flat)
          - Classifieur discr → Ward     (pm_disc)
        """
        U = np.asarray(U, dtype=float)
        n = len(U)
        k = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        # Signal comportement du classifieur (calculé par V5.4 avant
        # d'appeler super(), donc déjà disponible dans self._v54_u_flat)
        u_flat           = float(getattr(self, "_v54_u_flat", float(np.mean(U))))
        classifier_is_flat = u_flat >= self.u_flat_trigger

        # ── Choix du régime ─────────────────────────────────────
        use_ward = (
            self.ward_enabled
            and _HAS_AGGLO
            and not classifier_is_flat     # LR-like → Ward
        )
        pm = self.pm_flat if classifier_is_flat else self.pm_disc
        self._v55_clustering_used = "kmeans" if not use_ward else "ward"
        self._v55_pm_used         = pm

        if self.debug_clustering:
            print(
                f"[V55-cluster] u_flat={u_flat:.3f}  "
                f"flat={classifier_is_flat}  "
                f"→ {'KMeans' if not use_ward else 'Ward'}  pm={pm}"
            )

        # ── Pool top-N ──────────────────────────────────────────
        n_top  = min(n, max(k, k * pm))
        top    = np.argsort(-U)[:n_top]
        X_top  = X_unlabeled[top]
        nc     = min(k, len(X_top))

        if nc <= 1:
            return top[:k]

        # ── Clustering ──────────────────────────────────────────
        cv    = float(np.std(U) / (np.mean(U) + EPS))
        alpha = float(np.clip(0.6 - cv * 0.5, 0.1, 0.6))

        if use_ward:
            # ── Ward : partitions géométriques uniformes ─────────
            try:
                agglo  = AgglomerativeClustering(
                    n_clusters=nc, linkage=self.ward_linkage
                )
                labels = agglo.fit_predict(X_top)
            except Exception:
                # Fallback KMeans si Ward échoue
                use_ward = False

        if not use_ward:
            # ── KMeans U²-pondéré (WALC standard) ───────────────
            from sklearn.cluster import KMeans
            w  = np.power(U[top] + EPS, 2.0)
            km = KMeans(
                n_clusters=nc,
                n_init=self.n_init_kmeans,
                random_state=self.random_state,
            ).fit(X_top, sample_weight=w)
            labels = km.labels_

            sel = []
            for c, ctr in enumerate(km.cluster_centers_):
                m = np.where(labels == c)[0]
                if not len(m):
                    continue
                Un = _safe_minmax_norm(U[top[m]], eps=self.eps)
                dn = _safe_minmax_norm(
                    np.linalg.norm(X_top[m] - ctr, axis=1), eps=self.eps
                )
                score = (1 - alpha) * Un + alpha * (1 - dn)
                sel.append(int(top[m[int(np.argmax(score))]]))
            return np.array(list(dict.fromkeys(sel))[:k], dtype=int)

        # ── Sélection intra-cluster Ward ─────────────────────────
        # Score : (1-alpha)*U + alpha*(1-dist_centroïde_Ward)
        sel = []
        for c in range(nc):
            m = np.where(labels == c)[0]
            if not len(m):
                continue
            ctr = X_top[m].mean(axis=0)
            Un  = _safe_minmax_norm(U[top[m]], eps=self.eps)
            dn  = _safe_minmax_norm(
                np.linalg.norm(X_top[m] - ctr, axis=1), eps=self.eps
            )
            score = (1 - alpha) * Un + alpha * (1 - dn)
            sel.append(int(top[m[int(np.argmax(score))]]))

        return np.array(list(dict.fromkeys(sel))[:k], dtype=int)
