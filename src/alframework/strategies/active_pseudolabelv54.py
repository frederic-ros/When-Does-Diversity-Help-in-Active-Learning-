# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV54
====================

V5.4 = V5.3 + mécanisme smart adaptatif pour les scénarios déséquilibrés.

Deux mécanismes ajoutés par rapport à V5.3 :

1. Repondération U par (imbalance × incertitude-classe)
   --------------------------------------------------
   Pour chaque classe c prédite dans le pool :
     w(c) = freq_weight(c) × uncertainty_weight(c)

   - freq_weight(c) = n_labeled / (n_classes × n_labeled_c)
     → boost les classes sous-représentées dans y_labeled

   - uncertainty_weight(c) = U_mean(c) / U_mean_max
     → ne boost PAS une classe rare si elle est déjà bien séparée
     → évite de gaspiller du budget sur une minorité déjà certaine

   w(c) clippé dans [w_min, w_max] pour stabilité.

   Déclenchement conditionnel (seuil imb_ratio) :
     - imb_ratio = max(n_c) / min(n_c) dans y_labeled
     - si imb_ratio < imb_trigger_ratio → mécanisme OFF (cas équilibré)
     - si n_labeled < n_classes × min_labels_per_class → mécanisme OFF

2. Garde post-hoc de couverture des classes prédites
   ---------------------------------------------------
   Après sélection du batch :
     - Identifier les classes prédites dans le pool mais ABSENTES du batch
     - Pour chaque classe manquante : remplacer le point de plus faible U
       dans le batch par le point le plus incertain de cette classe dans le pool

   La garde s'applique toujours (indépendamment du déclenchement smart),
   mais n'agit que si le batch ne couvre pas toutes les classes prédites.

Paramètres nouveaux
-------------------
smart_enabled : bool = True
    Active/désactive l'ensemble du mécanisme smart.

imb_trigger_ratio : float = 1.5
    Ratio max(n_c)/min(n_c) dans y_labeled à partir duquel la repondération
    U s'active. En dessous : les classes sont assez équilibrées, on ne touche
    pas à U.

min_labels_per_class : int = 3
    Nombre minimal de labels par classe pour activer la repondération.
    Si une classe n'a pas encore assez de labels, les poids sont instables.

w_min : float = 0.5
    Plancher du poids w(c). Empêche de trop réduire une classe.

w_max : float = 6.0
    Plafond du poids w(c). Empêche un emballement sur une classe très rare.

u_flat_trigger : float = 0.50
    Seuil sur mean(U) du pool pour détecter un classifieur "plat" (uncertain
    partout) vs "discriminant" (frontière déjà trouvée).

    mean(U) élevé  → classifieur plat  (ex: RF early AL, non-linéaire)
                     → smart PEUT s'activer si imbalance détectée
    mean(U) faible → classifieur discriminant (ex: LR, frontière linéaire)
                     → smart reste OFF (U déjà bien structuré, pas besoin)

    Ce signal est model-agnostic : il détecte le COMPORTEMENT des probabilités,
    pas le type de modèle. Valeur par défaut 0.50 sépare RF (0.40–0.90) et
    LR (0.10–0.50) avec ~90% de précision empirique.

guard_enabled : bool = True
    Active/désactive la garde post-hoc indépendamment de la repondération.

debug_smart : bool = False
    Affiche les poids w(c) calculés et les classes manquantes corrigées.
"""

from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv53 import ActivePseudoLabelV53
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


EPS = 1e-12


@register("ActivePseudoLabelV54")
class ActivePseudoLabelV54(ActivePseudoLabelV53):
    """V5.4 : V5.3 + repondération U adaptative + garde post-hoc classes."""

    def __init__(
        self,
        *,
        # ── Mécanismes smart ──────────────────────────────────────
        smart_enabled: bool = True,
        imb_trigger_ratio: float = 2.0,
        min_labels_per_class: int = 3,
        w_min: float = 0.5,
        w_max: float = 6.0,
        u_flat_trigger: float = 0.50,
        guard_enabled: bool = True,
        debug_smart: bool = False,

        # ── Héritage V5.3 (pass-through complet) ─────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.smart_enabled        = bool(smart_enabled)
        self.imb_trigger_ratio    = float(imb_trigger_ratio)
        self.min_labels_per_class = int(min_labels_per_class)
        self.w_min                = float(w_min)
        self.w_max                = float(w_max)
        self.u_flat_trigger       = float(u_flat_trigger)
        self.guard_enabled        = bool(guard_enabled)
        self.debug_smart          = bool(debug_smart)

        # Diagnostics
        self._v54_imb_ratio         = 1.0
        self._v54_u_flat            = 0.0   # mean(U) du pool courant
        self._v54_smart_active      = False
        self._v54_weights           : dict = {}
        self._v54_guard_corrections : int  = 0

    # ------------------------------------------------------------------
    # Override du select complet pour capturer proba_u / y_l en cache
    # ------------------------------------------------------------------
    def select(self, state, budget: int) -> np.ndarray:
        """
        On surcharge select() pour mettre en cache proba_u et y_l,
        dont _select_diverse_from_top_adaptive a besoin.
        Le reste du pipeline V5.3 est inchangé.
        """
        import numpy as np
        self._v54_proba_u_cache = None
        self._v54_yl_cache      = None

        # Appel parent — il appellera _select_diverse_from_top_adaptive
        # via le chemin normal. On intercepte juste les données nécessaires.
        if hasattr(state, "y_labeled") and state.y_labeled is not None:
            self._v54_yl_cache = np.asarray(state.y_labeled)

        return super().select(state, budget)

    # ------------------------------------------------------------------
    # Override du _select_diverse_from_top_adaptive
    # ------------------------------------------------------------------
    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        V5.3 enrichi de :
          1. repondération U par imbalance × incertitude-classe (conditionnelle)
          2. garde post-hoc de couverture des classes prédites
        """
        U = np.asarray(U, dtype=float)
        n = len(U)
        k = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        # ── Récupération des données nécessaires ───────────────────
        y_l = self._v54_yl_cache

        # Prédictions brutes sur le pool (nécessaires pour les deux mécanismes)
        # On utilise le modèle mis en cache par V5.3 via _honest_model_ref
        model = getattr(self, "_honest_model_ref", None)
        proba_pool = None
        pred_pool  = None

        if model is not None and (self.smart_enabled or self.guard_enabled):
            try:
                proba_pool = model.predict_proba(X_unlabeled)
                pred_pool  = np.argmax(proba_pool, axis=1)
            except Exception:
                proba_pool = None
                pred_pool  = None

        # ── Signal comportement du classifieur : mean(U) ───────────
        # mean(U) élevé → classifieur plat (RF-like) → smart utile
        # mean(U) faible → classifieur discriminant (LR-like) → smart OFF
        u_flat = float(np.mean(U))
        self._v54_u_flat = u_flat
        classifier_is_flat = u_flat >= self.u_flat_trigger

        # ── 1. Repondération U ────────────────────────────────────
        U_adj = U.copy()

        self._v54_smart_active = False
        self._v54_weights = {}
        self._v54_imb_ratio = 1.0

        if (
            self.smart_enabled
            and classifier_is_flat          # classifieur plat (RF-like)
            and y_l is not None
            and len(y_l) > 0
            and proba_pool is not None
            and pred_pool  is not None
        ):
            classes, counts = np.unique(y_l, return_counts=True)
            n_classes = len(classes)
            n_tot     = len(y_l)

            # Ratio de déséquilibre dans y_labeled
            imb_ratio_labeled = float(max(counts)) / float(max(int(min(counts)), 1))

            # Ratio de déséquilibre dans les prédictions du pool
            # (plus représentatif dès le début avec init stratifiée)
            pool_preds = pred_pool
            _, pool_counts = np.unique(pool_preds, return_counts=True)
            imb_ratio_pool = (
                float(max(pool_counts)) / float(max(int(min(pool_counts)), 1))
                if len(pool_counts) > 1 else 1.0
            )

            # On prend le max des deux signaux
            imb_ratio = max(imb_ratio_labeled, imb_ratio_pool)
            self._v54_imb_ratio = imb_ratio

            # Conditions de déclenchement
            enough_labels = n_tot >= n_classes * self.min_labels_per_class
            imbalanced    = imb_ratio >= self.imb_trigger_ratio

            if imbalanced and enough_labels:
                self._v54_smart_active = True
                freq = dict(zip(classes.tolist(), counts.tolist()))

                # Incertitude marginale brute (non-normalisée) par classe prédite
                U_margin = self._base_uncertainty(proba_pool)   # appel interne V5.x
                u_per_cls = {}
                for c in classes:
                    mask = pred_pool == c
                    u_per_cls[c] = float(U_margin[mask].mean()) if mask.sum() > 0 else 0.0

                u_max = max(u_per_cls.values()) if u_per_cls else 1.0
                if u_max < EPS:
                    u_max = 1.0

                w_dict = {}
                for c in classes:
                    c = int(c)
                    freq_w = n_tot / (n_classes * (freq.get(c, 1) + EPS))
                    sep_w  = u_per_cls.get(c, 0.0) / u_max
                    w_c    = float(np.clip(freq_w * sep_w, self.w_min, self.w_max))
                    w_dict[c] = w_c
                    mask = pred_pool == c
                    U_adj[mask] *= w_c

                self._v54_weights = w_dict

                if self.debug_smart:
                    print(
                        f"[V54-smart] imb_ratio={imb_ratio:.2f}  "
                        f"u_flat={u_flat:.3f}  "
                        f"classes={n_classes}  n_tot={n_tot}"
                    )
                    for c, w in w_dict.items():
                        print(
                            f"  classe {c}: freq={freq.get(c,0)}  "
                            f"u_mean={u_per_cls.get(c,0.):.4f}  "
                            f"w={w:.3f}"
                        )

                # Re-normaliser U_adj dans [0,1] pour que le KMeans reste stable
                U_adj = _safe_minmax_norm(U_adj, eps=self.eps)

        # ── 2. Sélection V5.3 standard sur U_adj ──────────────────
        batch = super()._select_diverse_from_top_adaptive(
            X_unlabeled, U_adj, budget, k_eff
        )

        # ── 3. Garde post-hoc de couverture des classes ───────────
        # Gater derrière classifier_is_flat : sur un classifieur discriminant
        # (LR), les prédictions de classes en early AL sont peu fiables,
        # le guard introduirait du bruit inutile.
        self._v54_guard_corrections = 0

        if (
            self.guard_enabled
            and classifier_is_flat          # même condition que repondération
            and pred_pool is not None
            and len(batch) > 0
        ):
            batch_list = list(batch)
            batch_set  = set(batch_list)

            classes_pool  = set(np.unique(pred_pool).tolist())
            classes_batch = set(pred_pool[batch_list].tolist())
            missing       = classes_pool - classes_batch

            if missing:
                # Score de base (U non-pondéré) pour arbitrer les remplacements
                U_base_guard = _safe_minmax_norm(
                    self._base_uncertainty(proba_pool), eps=self.eps
                )
                scores_batch = U_base_guard[batch_list]
                replace_order = list(np.argsort(scores_batch))  # du + faible au + fort

                n_replaced = 0
                for cls in sorted(missing):
                    if n_replaced >= len(replace_order):
                        break
                    cands = [
                        c for c in np.where(pred_pool == cls)[0].tolist()
                        if c not in batch_set
                    ]
                    if not cands:
                        continue
                    best_cand = cands[int(np.argmax(U_base_guard[cands]))]
                    old_idx   = batch_list[replace_order[n_replaced]]
                    batch_list[batch_list.index(old_idx)] = best_cand
                    batch_set.discard(old_idx)
                    batch_set.add(best_cand)
                    n_replaced += 1

                self._v54_guard_corrections = n_replaced

                if self.debug_smart and n_replaced > 0:
                    print(
                        f"[V54-guard] {n_replaced} remplacement(s) "
                        f"pour classes manquantes: {sorted(missing)}"
                    )

                batch = np.asarray(batch_list[: int(budget)], dtype=int)

        return batch
