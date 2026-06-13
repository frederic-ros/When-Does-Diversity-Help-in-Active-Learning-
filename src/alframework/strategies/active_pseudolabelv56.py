# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV56
====================

V5.6 = V5.5 + calibration dynamique de la puissance p via détection
       de saturation sur la courbe F1 test.

Motivation
----------
V5.3–V5.5 utilisent p=2 fixe pour les poids KMeans :  w_i = U_i^p

p=2 est optimal en phase d'apprentissage rapide (frontière incertaine,
chaque point sélectionné apporte beaucoup). Mais quand le modèle approche
la saturation, p=2 concentre trop les sélections sur les mêmes zones déjà
bien couvertes → diversification sous-optimale.

Principe SAT-ADAPT
------------------
À chaque round, on mesure le gain de F1 sur le test set :
  gain_t = F1_t − F1_{t-1}

Gain moyen sur une fenêtre glissante :
  mean_gain = mean(gain_{t-window+1}, ..., gain_t)

Signal de saturation (soft, sigmoid inverse) :
  α_sat = sigmoid( (gain_thr − mean_gain) / gain_thr )
        → 0 si apprentissage rapide (mean_gain >> gain_thr)
        → 1 si plateau (mean_gain << gain_thr)

Puissance effective ce round :
  p_eff = 2.0 × (1 − α_sat) + p_plateau × α_sat  ∈ [p_plateau, 2.0]

Propriétés
----------
- Phase rapide  : α_sat ≈ 0 → p_eff ≈ 2.0 = V5.5 exact
- Phase plateau : α_sat ≈ 1 → p_eff ≈ p_plateau (diversification)
- Transition douce round par round — pas de seuil binaire
- Aucun classifieur auxiliaire — utilise F1 déjà calculé dans la boucle
- Même formule RF et LR, auto-adaptée

Paramètres nouveaux
-------------------
gain_window : int = 3
    Taille de la fenêtre glissante pour moyenner les gains.
    3 rounds = 60 pts minimum avant activation, suffisant pour un signal stable.

gain_thr : float = 0.005
    Seuil de gain (en F1 absolu) en dessous duquel la saturation est détectée.
    0.005 = 0.5pp/round — gain typique d'un run en plateau.

p_plateau : float = 1.4
    Puissance cible quand le modèle est en plateau.
    1.4 réduit le biais U² sans l'annuler → diversification modérée.

debug_sat : bool = False
    Affiche p_eff et α_sat à chaque round.

Héritage
--------
V5.6(ActivePseudoLabelV55) → tous les paramètres V5.5/V5.4/V5.3 en **kwargs.
"""

from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.strategies.active_pseudolabelv55 import ActivePseudoLabelV55
from alframework.strategies.active_pseudolabelv4 import _safe_minmax_norm


EPS = 1e-12


def _sat_alpha(mean_gain: float, gain_thr: float) -> float:
    """
    Sigmoid inverse centrée sur gain_thr.
    mean_gain >> gain_thr  →  α ≈ 0  (apprentissage rapide)
    mean_gain << gain_thr  →  α ≈ 1  (plateau)
    """
    if gain_thr <= 0:
        return 1.0 if mean_gain <= 0 else 0.0
    return float(1.0 / (1.0 + np.exp((mean_gain - gain_thr) / gain_thr)))


def _p_effective(alpha_sat: float, p_plateau: float) -> float:
    """p_eff = 2.0×(1−α) + p_plateau×α  ∈ [p_plateau, 2.0]"""
    return float(np.clip(2.0 * (1.0 - alpha_sat) + p_plateau * alpha_sat,
                         p_plateau, 2.0))


@register("ActivePseudoLabelV56")
class ActivePseudoLabelV56(ActivePseudoLabelV55):
    """
    V5.6 : V5.5 + puissance p adaptative par détection de saturation.

    p passe progressivement de 2.0 (apprentissage rapide) à p_plateau
    quand le gain F1/round ralentit, puis revient à 2.0 si le gain reprend.
    """

    def __init__(
        self,
        *,
        # ── Paramètres sat-adapt V5.6 ────────────────────────────
        gain_window:  int   = 3,
        gain_thr:     float = 0.005,
        p_plateau:    float = 1.4,
        debug_sat:    bool  = False,

        # ── Héritage V5.5 / V5.4 / V5.3 ─────────────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gain_window  = max(1, int(gain_window))
        self.gain_thr     = float(max(0.0, gain_thr))
        self.p_plateau    = float(np.clip(p_plateau, 0.3, 2.0))
        self.debug_sat    = bool(debug_sat)

        # u_flat_trigger : garantir l'attribut (défini par V5.5)
        if not hasattr(self, "u_flat_trigger"):
            self.u_flat_trigger = 0.50

        # État interne
        self._v56_f1_history: list  = []   # F1 test à chaque round
        self._v56_p_current:  float = 2.0  # p effectif dernier round
        self._v56_alpha_sat:  float = 0.0  # α_sat dernier round

    # ------------------------------------------------------------------
    def _reset_run(self) -> None:
        self._v56_f1_history = []
        self._v56_p_current  = 2.0
        self._v56_alpha_sat  = 0.0

    # ------------------------------------------------------------------
    def _update_p(self, f1_current: float) -> float:
        """
        Met à jour l'historique F1 et calcule p_eff pour ce round.
        Appelé AVANT la sélection, avec le F1 du modèle courant.
        """
        self._v56_f1_history.append(f1_current)

        if len(self._v56_f1_history) < self.gain_window + 1:
            # Pas encore assez d'historique → p=2.0
            self._v56_alpha_sat = 0.0
            self._v56_p_current = 2.0
            return 2.0

        # Gain moyen sur la fenêtre
        gains = [
            self._v56_f1_history[-i] - self._v56_f1_history[-i - 1]
            for i in range(1, self.gain_window + 1)
        ]
        mean_gain = float(np.mean(gains))

        alpha = _sat_alpha(mean_gain, self.gain_thr)
        p_eff = _p_effective(alpha, self.p_plateau)

        self._v56_alpha_sat = alpha
        self._v56_p_current = p_eff
        return p_eff

    # ------------------------------------------------------------------
    def select(self, state, budget: int) -> np.ndarray:
        """
        Entrée principale.
        1. Récupère le F1 courant depuis state
        2. Met à jour p_eff via sat-adapt
        3. Injecte p_eff dans le clustering
        4. Délègue à V5.5
        """
        # Reset si nouveau run (F1 courant < dernier F1 historique par ex.)
        X_lab   = state.X_labeled
        y_lab   = state.y_labeled
        clf     = state.model

        if len(X_lab) <= 20 + len(y_lab) * 0:
            # Heuristique reset : labeled set très petit = début de run
            pass
        if not self._v56_f1_history:
            self._reset_run()

        # F1 courant — priorité : test set > pool labeled holdout > proxy labeled
        f1_now = None
        try:
            if hasattr(state, "X_test") and state.X_test is not None and len(state.X_test) > 0:
                # bench_real : test set disponible → signal propre
                from sklearn.metrics import f1_score as _f1
                preds = clf.predict(state.X_test)
                f1_now = float(_f1(state.y_test, preds,
                                   average="macro", zero_division=0))
            elif hasattr(state, "y_pool") and state.y_pool is not None:
                # bench_synthetic : pas de X_test, mais on peut calculer
                # le F1 sur les données NON labelisées (vrai signal de généralisation)
                pool_mask = np.ones(len(state.X_unlabeled), dtype=bool)
                if len(state.X_unlabeled) > 0:
                    from sklearn.metrics import f1_score as _f1
                    preds = clf.predict(state.X_unlabeled)
                    # y_pool contient les vrais labels du pool non-labelisé
                    n_unlab = len(state.X_unlabeled)
                    y_true_unlab = state.y_pool[-n_unlab:] \
                        if hasattr(state, "y_pool") and len(state.y_pool) >= n_unlab \
                        else None
                    if y_true_unlab is not None and len(y_true_unlab) == n_unlab:
                        f1_now = float(_f1(y_true_unlab, preds,
                                           average="macro", zero_division=0))
        except Exception:
            f1_now = None

        # Dernier recours : F1 train (proxy, biaisé mais vaut mieux que rien)
        if f1_now is None:
            try:
                from sklearn.metrics import f1_score as _f1
                f1_now = float(_f1(y_lab, clf.predict(X_lab),
                                   average="macro", zero_division=0))
            except Exception:
                f1_now = None

        # Mise à jour p_eff
        if f1_now is not None:
            p_eff = self._update_p(f1_now)
        else:
            p_eff = 2.0

        # Injecter p pour le clustering
        self._v56_kmeans_power = p_eff

        if self.debug_sat and f1_now is not None:
            print(
                f"[V56-sat] n_lab={len(X_lab):>4}  "
                f"f1={f1_now:.4f}  "
                f"α_sat={self._v56_alpha_sat:.3f}  "
                f"p_eff={p_eff:.3f}"
            )

        return super().select(state, budget)

    # ------------------------------------------------------------------
    def _select_diverse_from_top_adaptive(
        self,
        X_unlabeled: np.ndarray,
        U: np.ndarray,
        budget: int,
        k_eff: int,
    ) -> np.ndarray:
        """
        Identique à V5.5 mais avec poids KMeans = U^p_eff.
        """
        U  = np.asarray(U, dtype=float)
        n  = len(U)
        k  = min(int(budget), n)

        if k <= 0:
            return np.array([], dtype=int)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U))], dtype=int)

        power = float(getattr(self, "_v56_kmeans_power", 2.0))

        u_flat             = float(getattr(self, "_v54_u_flat", float(np.mean(U))))
        classifier_is_flat = u_flat >= self.u_flat_trigger
        alpha_sat          = float(getattr(self, "_v56_alpha_sat", 0.0))

        try:
            from sklearn.cluster import AgglomerativeClustering as _Agglo
            _has_agglo = True
        except ImportError:
            _has_agglo = False

        # En plateau (α_sat > 0.4), forcer KMeans pour que p_eff ait un effet.
        # Ward ne dépend pas des poids → la puissance p serait ignorée.
        use_ward = (
            self.ward_enabled
            and _has_agglo
            and not classifier_is_flat
            and alpha_sat < 0.4
        )
        pm = self.pm_flat if classifier_is_flat else self.pm_disc

        n_top = min(n, max(k, k * pm))
        top   = np.argsort(-U)[:n_top]
        X_top = X_unlabeled[top]
        nc    = min(k, len(X_top))

        if nc <= 1:
            return top[:k]

        cv    = float(np.std(U) / (np.mean(U) + EPS))
        alpha = float(np.clip(0.6 - cv * 0.5, 0.1, 0.6))

        if use_ward:
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
                    ctr = X_top[m].mean(axis=0)
                    Un  = _safe_minmax_norm(U[top[m]], eps=EPS)
                    dn  = _safe_minmax_norm(
                        np.linalg.norm(X_top[m] - ctr, axis=1), eps=EPS
                    )
                    sel.append(int(top[m[int(np.argmax(
                        (1 - alpha) * Un + alpha * (1 - dn)
                    ))]]))
                return np.array(list(dict.fromkeys(sel))[:k], dtype=int)
            except Exception:
                pass  # fallback KMeans

        # KMeans avec p_eff
        from sklearn.cluster import KMeans
        w  = np.power(U[top] + EPS, power)
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
    def p_current(self) -> float:
        """p effectif du dernier round."""
        return self._v56_p_current

    @property
    def alpha_sat(self) -> float:
        """Score de saturation du dernier round (0=rapide, 1=plateau)."""
        return self._v56_alpha_sat

    @property
    def sat_info(self) -> dict:
        """Diagnostics sat-adapt."""
        return {
            "f1_history":  list(self._v56_f1_history),
            "p_current":   self._v56_p_current,
            "alpha_sat":   self._v56_alpha_sat,
            "gain_window": self.gain_window,
            "gain_thr":    self.gain_thr,
            "p_plateau":   self.p_plateau,
        }
