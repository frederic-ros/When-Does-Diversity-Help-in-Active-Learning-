# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV52
====================

V52 = union simple V4.4 ↔ V5.1, sans nouvelle couche complexe.

Idée
----
V4.4 et V5.1 sont complémentaires :
- V4.4 est meilleur quand le signal actif est exploitable / contrasté.
- V5.1 est meilleur quand le signal est plat, bruité, déséquilibré ou ambigu.

V52 ne fait pas un gros ensemble coûteux. Elle garde exactement le backbone V5.1
(V4.4 + alpha controller léger), mais ajoute un routeur très simple :

    mode = "V44"  -> alpha V4.4 pur
    mode = "V51"  -> alpha V5.1 corrigé

Le routing se fait au niveau de alpha, donc le coût reste identique à V5.1.
Pas de double KMeans. Pas de double sélection. Pas de densité locale.

Règles par défaut
-----------------
1. Si lam_eff est suffisant ET le score U est contrasté : faire confiance à V4.4.
2. Si haute dimension forte ET signal non plat : revenir à V4.4, car les centroïdes
   deviennent moins fiables.
3. Sinon utiliser V5.1, qui hedge mieux les cas plats / difficiles.

Pourquoi pas un vrai ensemble ?
-------------------------------
Un vrai ensemble V4.4 + V5.1 nécessiterait de calculer deux sélections puis fusionner.
Cela doublerait quasiment le coût et compliquerait le comportement. V52 vise le même
bénéfice empirique, mais avec une règle lisible et quasi gratuite.
"""

from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv51 import ActivePseudoLabelV51


@register("ActivePseudoLabelV52")
class ActivePseudoLabelV52(ActivePseudoLabelV51):
    """
    V52 : routeur léger entre V4.4 et V5.1.

    Paramètres ajoutés
    ------------------
    router_enabled : bool, default True
        Active le routeur V44/V51. Si False, comportement identique à V5.1.

    v44_lam_min : float, default 0.12
        Lam_eff minimal pour considérer que le signal V4.4 est assez fiable.

    v44_contrast_min : float, default 0.30
        Contraste minimal du score U pour repasser en mode V4.4 pur.

    highdim_v44_min : float, default 0.65
        Pression haute dimension à partir de laquelle on préfère V4.4 si le
        signal n'est pas trop plat.

    highdim_flatness_max : float, default 0.85
        En haute dimension, si U est extrêmement plat, on garde V5.1 malgré tout.

    debug_router : bool, default False
        Si True, stocke les raisons de routing dans self._v52_router_reason.

    Hérite de tous les paramètres V5.1 :
    - lam_threshold
    - flatness_alpha_weight
    - contrast_alpha_weight
    - imbalance_alpha_weight
    - highdim_alpha_weight
    - etc.
    """

    def __init__(
    self,
    *,
    router_enabled: bool = True,
    v44_lam_min: float = 0.12,
    v44_contrast_min: float = 0.30,
    highdim_v44_min: float = 0.65,
    highdim_flatness_max: float = 0.85,
    debug_router: bool = False,

    # aliases panel
    contrast_threshold: float | None = None,
    flatness_threshold: float | None = None,
    highdim_threshold: float | None = None,

    **kwargs,
):
        if contrast_threshold is not None:
            v44_contrast_min = float(contrast_threshold)

        if highdim_threshold is not None:
            highdim_v44_min = float(highdim_threshold)

        # flatness_threshold est accepté pour éviter l'erreur init,
        # mais cette version V52 utilise highdim_flatness_max comme garde.
        if flatness_threshold is not None:
            highdim_flatness_max = float(max(highdim_flatness_max, flatness_threshold))

        super().__init__(**kwargs)

        self.router_enabled = bool(router_enabled)
        self.v44_lam_min = float(v44_lam_min)
        self.v44_contrast_min = float(v44_contrast_min)
        self.highdim_v44_min = float(highdim_v44_min)
        self.highdim_flatness_max = float(highdim_flatness_max)
        self.debug_router = bool(debug_router)

        self._v52_mode = "V51"
        self._v52_router_reason = "default"       

    def _effective_alpha_v51(
        self,
        *,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
    ) -> float:
        """
        Routeur V52.

        On calcule d'abord les diagnostics et alpha_base V4.4.
        Ensuite :
        - si le régime ressemble à un bon régime V4.4, retourner alpha_base ;
        - sinon, retourner alpha V5.1 via la méthode parente.
        """
        # Alpha V4.4 pur
        alpha_base = self._alpha_from_lam_eff()
        self._v51_alpha_base = float(alpha_base)

        if not self.router_enabled:
            self._v52_mode = "V51"
            self._v52_router_reason = "router_disabled"
            return super()._effective_alpha_v51(X_unlabeled=X_unlabeled, U=U)

        # Diagnostics V5.1, réutilisés sans nouvelle complexité
        flatness = self._score_flatness(U)
        contrast = self._score_contrast(U)
        highdim = self._highdim_pressure(X_unlabeled)
        imbalance = self._labeled_imbalance_from_cache()
        lam_eff = float(getattr(self, "_last_lam_eff", 0.0))

        self._v51_flatness = flatness
        self._v51_contrast = contrast
        self._v51_highdim = highdim
        self._v51_imbalance = imbalance

        # --------------------------------------------------------------
        # Route 1 : signal actif exploitable -> V4.4 pur
        # --------------------------------------------------------------
        # Cas typique : easy / redundancy / propagation utile.
        # V5.1 peut sur-corriger alpha ; ici on garde alpha V4.4.
        if lam_eff >= self.v44_lam_min and contrast >= self.v44_contrast_min:
            alpha = float(np.clip(alpha_base, self.alpha_min, self.alpha_max))
            self._v51_alpha_final = alpha
            self._v52_mode = "V44"
            self._v52_router_reason = "strong_lam_and_contrast"
            return alpha

        # --------------------------------------------------------------
        # Route 2 : haute dimension + signal pas totalement plat -> V4.4
        # --------------------------------------------------------------
        # Les pertes V5.1 observées en highdim suggèrent de réduire la
        # correction centroïde dans ces cas. Si U est très plat, on garde V5.1.
        if highdim >= self.highdim_v44_min and flatness <= self.highdim_flatness_max:
            alpha = float(np.clip(alpha_base, self.alpha_min, self.alpha_max))
            self._v51_alpha_final = alpha
            self._v52_mode = "V44"
            self._v52_router_reason = "highdim_guard"
            return alpha

        # --------------------------------------------------------------
        # Sinon : V5.1, meilleur hedge dans les cas ambigus / plats
        # --------------------------------------------------------------
        self._v52_mode = "V51"
        self._v52_router_reason = "weak_or_flat_signal"
        return super()._effective_alpha_v51(X_unlabeled=X_unlabeled, U=U)
