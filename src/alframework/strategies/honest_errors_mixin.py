# -*- coding: utf-8 -*-
"""
honest_errors_mixin.py
======================

Backbone-agnostic fix for the "propagation never activates" problem.

Diagnosis (reproduced on CIFAR-10 PCA100, strate=5000, RF-200):
    src_n    = 0   at every iteration
    gate     = 0.0 at every iteration
    lam_eff  = 0.0 at every iteration  (0/30 rounds)

Root cause
----------
`_source_scores` derives propagation sources from the *labeled points the
model misclassifies*:

    pred_l = argmax(model.predict_proba(X_labeled))
    err_mask = pred_l != y_labeled            # ALWAYS False for RF-200
    err_indices = where(err_mask)             # ALWAYS empty

A RandomForest with 200 trees memorises its training set perfectly, so
the model never "errs" on X_labeled. With no error sources, U_prop = 0,
gate = 0, lam_eff = 0, and the whole V4 family collapses to its
geometric / uncertainty fallback. The propagation mechanism — the core
claim of the paper — is inert.

Fix (generic, backbone-agnostic)
--------------------------------
Replace the memorised `proba_labeled` by an HONEST out-of-fold estimate
obtained with `cross_val_predict`, which works for ANY scikit-learn
estimator exposing predict_proba (RF, LogisticRegression, SVC(probability),
GradientBoosting, ...). When predict_proba is unavailable, fall back to a
hard-label `cross_val_predict` and synthesise a one-hot proba matrix.

Cold-start safety
-----------------
Cross-validation needs at least `cv` samples per class. In active learning
the seed set can be tiny (e.g. 10 labels / 10 classes). The mixin:
  * computes the minimum per-class count,
  * adapts cv = min(cv_max, min_per_class),
  * if cv < 2 (a class has a single sample), it CANNOT build an honest
    estimate -> it returns the memorised proba unchanged, so behaviour is
    *identical to the legacy code* in that regime (lam_eff stays 0, no
    regression vs previously reported cold-start numbers).

Opt-in
------
`honest_errors=False` by default -> strictly legacy behaviour, zero impact
on any previously reported result. Set `honest_errors=True` to activate.

Usage
-----
Make a V4-family class inherit the mixin BEFORE the V4 base, e.g.:

    class ActivePseudoLabelV4H(HonestErrorsMixin, ActivePseudoLabelV4):
        pass

and register it. The mixin only overrides `_honest_proba_labeled` and
wraps `_source_scores` to substitute proba_labeled when appropriate.
Nothing else in the V4 pipeline is touched.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class HonestErrorsMixin:
    """
    Mixin that substitutes a cross-validated (honest) proba_labeled into
    `_source_scores`, so that propagation sources can actually be detected
    even with a perfectly-memorising backbone (RandomForest, etc.).

    New parameters (all optional, legacy-safe defaults)
    ---------------------------------------------------
    honest_errors : bool, default False
        Master switch. False -> legacy behaviour (no change at all).

    honest_cv_max : int, default 5
        Upper bound for the number of CV folds. Effective cv is
        min(honest_cv_max, min_samples_per_class).

    honest_min_cv : int, default 2
        Minimum folds required to attempt an honest estimate. If the
        smallest class has fewer than this many samples, the mixin
        returns the memorised proba unchanged (legacy fallback).

    honest_random_state : int, default 0
        Seed for the StratifiedKFold used by cross_val_predict.
    """

    def __init__(
        self,
        *,
        honest_errors: bool = False,
        honest_cv_max: int = 5,
        honest_min_cv: int = 2,
        honest_random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.honest_errors = bool(honest_errors)
        self.honest_cv_max = int(honest_cv_max)
        self.honest_min_cv = int(honest_min_cv)
        self.honest_random_state = int(honest_random_state)
        # Diagnostics (inspectable after each select())
        self._honest_last_cv = 0
        self._honest_last_active = False

    # ------------------------------------------------------------------
    # Honest proba estimation
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
        Return an out-of-fold proba matrix aligned on `classes_seen`
        (same column layout as the legacy proba_labeled), or the legacy
        matrix unchanged if an honest estimate cannot be built safely.
        """
        self._honest_last_active = False
        self._honest_last_cv = 0

        if not self.honest_errors:
            return proba_labeled_legacy

        n_l = len(X_labeled)
        if n_l == 0:
            return proba_labeled_legacy

        y_labeled = np.asarray(y_labeled)
        uniq, counts = np.unique(y_labeled, return_counts=True)

        # Need >= 2 distinct classes for a meaningful CV error signal.
        if uniq.size < 2:
            return proba_labeled_legacy

        min_per_class = int(counts.min())
        cv = min(int(self.honest_cv_max), min_per_class)

        # Cold-start / degenerate: cannot build honest folds -> legacy.
        if cv < int(self.honest_min_cv):
            return proba_labeled_legacy

        # Column layout to match the legacy matrix / classes_seen.
        if classes_seen is None:
            classes_seen = np.unique(y_labeled)
        classes_seen = np.asarray(classes_seen)
        col_of = {int(c): j for j, c in enumerate(classes_seen)}

        try:
            skf = StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.honest_random_state,
            )
            est = clone(model)

            has_proba = hasattr(est, "predict_proba")
            if has_proba:
                oof = cross_val_predict(
                    est, X_labeled, y_labeled,
                    cv=skf, method="predict_proba",
                )
                # cross_val_predict orders columns by np.unique(y) of the
                # estimator's classes_; remap onto classes_seen layout.
                fitted_classes = np.unique(y_labeled)
                proba = np.zeros((n_l, len(classes_seen)), dtype=float)
                for src_col, c in enumerate(fitted_classes):
                    c = int(c)
                    if c in col_of:
                        proba[:, col_of[c]] = oof[:, src_col]
            else:
                # Hard-label fallback -> one-hot honest "proba".
                oof_lab = cross_val_predict(
                    est, X_labeled, y_labeled, cv=skf, method="predict",
                )
                proba = np.zeros((n_l, len(classes_seen)), dtype=float)
                for row, c in enumerate(oof_lab):
                    c = int(c)
                    if c in col_of:
                        proba[row, col_of[c]] = 1.0

            # Row-normalise defensively (predict_proba should already sum 1).
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0.0] = 1.0
            proba = proba / row_sums

            self._honest_last_active = True
            self._honest_last_cv = cv
            return proba

        except Exception as e:
            # Any CV failure (e.g. a class vanished in a fold) -> legacy.
            print(f"[honest_errors] CV failed ({e!r}); using legacy proba.")
            return proba_labeled_legacy

    # ------------------------------------------------------------------
    # Wrap _source_scores: swap proba_labeled before the legacy logic.
    # ------------------------------------------------------------------

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
        if self.honest_errors:
            model = getattr(self, "_honest_model_ref", None)
            if model is not None:
                proba_labeled = self._honest_proba_labeled(
                    model=model,
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    proba_labeled_legacy=proba_labeled,
                    classes_seen=classes_seen,
                )
        return super()._source_scores(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            proba_labeled=proba_labeled,
            X_unlabeled=X_unlabeled,
            proba_unlabeled=proba_unlabeled,
            k_eff=k_eff,
            classes_seen=classes_seen,
        )

    # ------------------------------------------------------------------
    # Capture the model reference at the start of select().
    # ------------------------------------------------------------------

    def select(self, state, budget: int):
        # Stash the live estimator so _source_scores can clone it for CV.
        self._honest_model_ref = getattr(state, "model", None)
        return super().select(state, budget)
