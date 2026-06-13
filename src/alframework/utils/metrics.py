# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:11:36 2026

@author: frederic.ros
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import label_binarize


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


def confusion_derived_metrics(
    cm: np.ndarray,
) -> Dict[str, Any]:
    """
    Extrait des métriques par classe à partir d'une matrice de confusion multi-classe.
    Retourne:
      - support_per_class
      - recall_per_class (TPR)
      - precision_per_class
      - specificity_per_class (TNR)
      - fpr_per_class
      - fnr_per_class
      - macro_specificity, macro_fpr, macro_fnr
    """
    cm = np.asarray(cm, dtype=float)
    k = cm.shape[0]
    assert cm.shape == (k, k)

    tp = np.diag(cm)
    support = cm.sum(axis=1)  # vrais par classe
    pred_support = cm.sum(axis=0)  # prédits par classe
    total = cm.sum()

    fn = support - tp
    fp = pred_support - tp
    tn = total - (tp + fp + fn)

    recall = _safe_div(tp, tp + fn)              # TPR
    precision = _safe_div(tp, tp + fp)
    specificity = _safe_div(tn, tn + fp)         # TNR
    fpr = 1.0 - specificity
    fnr = 1.0 - recall

    return {
        "support_per_class": support.astype(int),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "specificity_per_class": specificity,
        "fpr_per_class": fpr,
        "fnr_per_class": fnr,
        "macro_specificity": float(np.mean(specificity)),
        "macro_fpr": float(np.mean(fpr)),
        "macro_fnr": float(np.mean(fnr)),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    labels: Optional[Sequence[Any]] = None,
    normalize_cm: Optional[str] = None,  # None | "true" | "pred" | "all"
) -> Dict[str, Any]:
    """
    Calcule un pack de métriques robustes pour multi-classe.
    - y_true, y_pred: shape (n,)
    - y_proba: shape (n, n_classes) optionnel
    - labels: fixe l'ordre des classes (recommandé si tu agrèges sur plein de bases)
    - normalize_cm: normalisation sklearn de la confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # labels: important pour comparabilité entre datasets
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred], axis=0))

    labels = list(labels)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    out: Dict[str, Any] = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels": labels,
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm

    # métriques dérivées de la CM (par classe + macro)
    out.update(confusion_derived_metrics(cm))

    # Métriques basées proba (si dispo)
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        # log_loss (cross-entropy) multi-classe
        # Important: labels=labels pour aligner colonnes proba avec ordre des classes
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        # Brier score multi-classe (moyenne MSE sur one-hot)
        Y = label_binarize(y_true, classes=labels)
        if Y.shape[1] == 0:  # cas extrême 1 classe
            out["brier_score"] = float("nan")
        else:
            out["brier_score"] = float(np.mean((Y - y_proba) ** 2))

    # Option: CM normalisée séparée si tu veux la sauver aussi
    if normalize_cm is not None:
        out["confusion_matrix_normalized"] = confusion_matrix(
            y_true, y_pred, labels=labels, normalize=normalize_cm
        )

    return out


def fit_predict_and_score(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    labels: Optional[Sequence[Any]] = None,
    proba: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne un classifieur puis calcule toutes les métriques sur le test.
    Utilise predict_proba si dispo et proba=True.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if proba and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    return compute_classification_metrics(y_test, y_pred, y_proba=y_proba, labels=labels)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

