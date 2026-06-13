from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np

# Métriques scalaires extraites après calcul complet
_SCALAR_KEYS = (
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted", "f1_micro",
    "precision_macro", "recall_macro",
    "precision_weighted", "recall_weighted",
    "macro_specificity", "macro_fpr", "macro_fnr",
    "log_loss", "brier_score",
)


def _align_and_normalize_proba(
    y_proba: Optional[np.ndarray],
    *,
    model_classes: Optional[Sequence] = None,
    labels: Optional[Sequence] = None,
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    """
    Sécurise les probabilités avant compute_classification_metrics / log_loss.

    Cas traités :
    - probabilités qui ne somment pas exactement à 1 ;
    - NaN / inf ;
    - modèle entraîné sur un sous-ensemble de classes : predict_proba retourne
      moins de colonnes que la liste globale `labels`.

    Si `labels` et `model_classes` sont disponibles, on réaligne les colonnes
    de y_proba sur l'ordre global de `labels`. Les classes absentes reçoivent
    une petite masse eps, puis chaque ligne est renormalisée.
    """
    if y_proba is None:
        return None

    P = np.asarray(y_proba, dtype=float)

    if P.ndim != 2 or P.shape[0] == 0:
        return None

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, eps, 1.0)

    if labels is not None and model_classes is not None:
        labels_arr = np.asarray(list(labels))
        classes_arr = np.asarray(list(model_classes))

        if len(labels_arr) > 0 and len(classes_arr) == P.shape[1]:
            aligned = np.full((P.shape[0], len(labels_arr)), eps, dtype=float)

            pos = {c: j for j, c in enumerate(labels_arr.tolist())}

            for src_j, c in enumerate(classes_arr.tolist()):
                dst_j = pos.get(c, None)
                if dst_j is not None:
                    aligned[:, dst_j] = P[:, src_j]

            P = aligned

    row_sum = P.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= eps, 1.0, row_sum)
    P = P / row_sum

    return P


def evaluate(
    model: Any,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    *,
    labels: Optional[Sequence] = None,
) -> Dict[str, float]:
    """
    Évalue le modèle sur (X_test, y_test) et retourne un dict de métriques scalaires.
    Utilise compute_classification_metrics (utils/metrics.py) pour la cohérence globale.
    Inclut : accuracy, balanced_accuracy, f1_macro/weighted/micro, précision/rappel,
             spécificité, FPR/FNR, log_loss, brier_score (si predict_proba dispo).
    """
    if X_test is None or y_test is None:
        return {}

    # Import local pour éviter la circularité core -> utils au niveau du module
    from alframework.utils.metrics import compute_classification_metrics  # noqa

    y_pred = model.predict(X_test)
    y_proba: Optional[np.ndarray] = None
    model_classes: Optional[Sequence] = getattr(model, "classes_", None)

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            y_proba = _align_and_normalize_proba(
                y_proba,
                model_classes=model_classes,
                labels=labels,
            )
        except Exception:
            y_proba = None

    all_m = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
    )

    # Ne garder que les scalaires float (pas les tableaux confusion_matrix, etc.)
    return {k: float(all_m[k]) for k in _SCALAR_KEYS if k in all_m and isinstance(all_m[k], (int, float, np.floating))}
