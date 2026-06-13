# -*- coding: utf-8 -*-
"""
tests/common.py
===============
Module partagé pour tous les scripts de test.

Centralise :
  - La résolution du sys.path (src/)
  - La purge des modules alframework (Spyder)
  - Les imports de stratégies (enregistrement dans le registry)
  - Les utilitaires partagés : _fit_eval, _validate_indices, ArrayLabeler
  - Les types partagés : CurvePoint, LearningCurveResult

Utilisation dans un script test :
    from common import *   # ou imports sélectifs
"""
from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import clone

# ---------------------------------------------------------------------------
# 1. Résolution du sys.path
# ---------------------------------------------------------------------------

def _setup_src_path() -> Path:
    """Ajoute src/ au sys.path depuis n'importe quel emplacement d'appel."""
    this_file = Path(__file__).resolve()
    project_root = this_file.parent
    while project_root != project_root.parent:
        src = project_root / "src"
        if src.exists():
            break
        project_root = project_root.parent
    else:
        raise RuntimeError(f"Impossible de trouver 'src/' depuis {this_file}")

    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def purge_alframework_modules() -> None:
    """
    Invalide les imports partiels d'alframework (utile sous Spyder quand on
    relance un script après modification du code source).
    """
    for k in list(sys.modules.keys()):
        if k == "alframework" or k.startswith("alframework."):
            del sys.modules[k]
    importlib.invalidate_caches()


# Exécution au moment de l'import du module
_setup_src_path()
purge_alframework_modules()

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak*")

# ---------------------------------------------------------------------------
# 2. Imports du framework + enregistrement des stratégies dans le registry
# ---------------------------------------------------------------------------
import alframework  # noqa: E402  (doit être APRÈS setup_src_path)

from alframework.core.registry import STRATEGIES  # noqa: E402
from alframework.core.state import ALState  # noqa: E402
from alframework.core.runner import active_learning_loop  # noqa: E402
from alframework.core.metrics import evaluate  # noqa: E402
from alframework.core.labeler import ArrayLabeler  # noqa: E402  -- source unique

from alframework.data.synth_builder import build_synth_state_and_testset  # noqa: E402
from alframework.utils.metrics import compute_classification_metrics  # noqa: E402
from alframework.utils.seed import seed_everything  # noqa: E402
from alframework.utils.curve_utils import (  # noqa: E402
    CurvePoint,
    auc_trapz,
    curve_to_arrays,
    make_budget_grid,
    normalize_auc,
)
from alframework.config.config_strategies import make_strategy, enabled_strategy_names  # noqa: E402
from alframework.config.strategies_parameter import (  # noqa: E402
    make_strategy_from_config,
    validate_strategy_config,
)

# -- Enregistrement de toutes les stratégies dans le registry --
# Ces imports ont un side-effect : ils appellent @register_strategy()
from alframework.strategies import bait_simple as _bait           # noqa: F401
from alframework.strategies import random as _random              # noqa: F401
from alframework.strategies import uncertainty as _unc            # noqa: F401
from alframework.strategies import coreset as _core               # noqa: F401
from alframework.strategies import typiclust as _typ              # noqa: F401
from alframework.strategies import probcover as _pc               # noqa: F401
from alframework.strategies import dbal as _dbal                  # noqa: F401
from alframework.strategies import badge as _badge                # noqa: F401
from alframework.strategies import rank2022 as _rank              # noqa: F401
from alframework.strategies import active_pseudolabel as _apl     # noqa: F401
from alframework.strategies import activepseudolabelv2 as _aplv2
from alframework.strategies import qbc as _qbc                    # noqa: F401
from alframework.strategies import tri_committee as _tri          # noqa: F401
from alframework.strategies import selftrain_acq as _sta          # noqa: F401
from alframework.strategies import robust_qbc as _rqbc            # noqa: F401
from alframework.strategies import adaptive_disagreement as _ads  # noqa: F401
from alframework.strategies import diversity_optimized_batch as _dobs  # noqa: F401
from alframework.strategies import active_pseudolabel_v3 as _aplv3
from alframework.strategies import active_pseudolabelv4 as _aplv4

# ---------------------------------------------------------------------------
# 3. Utilitaires partagés
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningCurveResult:
    """Résultat d'une courbe d'apprentissage pour une stratégie."""
    selected_indices: np.ndarray   # vide en mode itératif
    curve: List[CurvePoint]
    auc: Dict[str, float]
    auc_norm: Dict[str, float]
    params: Dict[str, Any]


def _fit_eval(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[Any],
) -> Dict[str, Any]:
    """Entraîne un clone du modèle et retourne toutes les métriques de classification."""
    m = clone(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
    return compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
        normalize_cm="true",
    )


def _validate_indices(
    idx: np.ndarray,
    n_unlabeled: int,
    max_budget: int,
    name: str,
) -> None:
    """Valide les indices retournés par une stratégie."""
    if idx.ndim != 1:
        raise ValueError(f"{name}: idx doit être 1D, got shape {idx.shape}")
    if len(idx) > max_budget:
        raise ValueError(f"{name}: idx retourne {len(idx)} > max_budget={max_budget}")
    if len(idx) == 0:
        return
    if (idx < 0).any():
        raise ValueError(f"{name}: indices négatifs détectés")
    if (idx >= n_unlabeled).any():
        raise ValueError(f"{name}: indices hors limites (max={n_unlabeled-1})")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name}: doublons dans idx")


def _instantiate_strategy(name: str, *, parameterconfig: bool, overrides: Optional[Dict] = None) -> Any:
    """Instancie une stratégie depuis le registry ou la config paramétrique."""
    overrides = overrides or {}
    if parameterconfig:
        return make_strategy_from_config(name, overrides=overrides, validate=True)
    cls = STRATEGIES[name]
    return cls(**overrides) if overrides else cls()


def _copy_state(state0: ALState, X_test: np.ndarray, y_test: np.ndarray) -> ALState:
    """Copie profonde d'un ALState pour isolation par stratégie en mode itératif."""
    new_state = ALState(
        X_labeled=state0.X_labeled.copy(),
        y_labeled=state0.y_labeled.copy(),
        X_unlabeled=state0.X_unlabeled.copy(),
        model=clone(state0.model),
        rng=state0.rng,
        X_test=X_test,
        y_test=y_test,
    )
    for attr in ("n_classes", "classes_all", "base_model"):
        if hasattr(state0, attr):
            setattr(new_state, attr, getattr(state0, attr))
    if hasattr(state0, "classes_all") and not hasattr(new_state, "labels"):
        new_state.labels = list(getattr(state0, "classes_all"))
    return new_state


def _compute_auc(curve: List[CurvePoint], tracked_metrics: List[str]) -> tuple:
    """Calcule AUC et AUC normalisée pour chaque métrique."""
    auc: Dict[str, float] = {}
    auc_norm: Dict[str, float] = {}
    for k in tracked_metrics:
        x, y = curve_to_arrays(curve, metric=k)
        a = auc_trapz(x, y)
        auc[k] = a
        auc_norm[k] = normalize_auc(a, float(x[0]), float(x[-1]))
    return auc, auc_norm


# Métriques par défaut pour les courbes d'apprentissage
DEFAULT_TRACKED_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]

# Métriques scalaires retournées par l'évaluation finale (audit)
DEFAULT_METRIC_KEYS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
DEFAULT_GAP_KEYS = ["accuracy_gap", "balanced_accuracy_gap", "f1_macro_gap"]

__all__ = [
    # setup
    "purge_alframework_modules",
    # framework
    "STRATEGIES", "ALState", "active_learning_loop", "evaluate", "ArrayLabeler",
    "build_synth_state_and_testset", "compute_classification_metrics",
    "make_strategy_from_config", "validate_strategy_config",
    "make_strategy", "enabled_strategy_names",
    # curve utils
    "CurvePoint", "auc_trapz", "curve_to_arrays", "make_budget_grid", "normalize_auc",
    # shared types & utils
    "LearningCurveResult",
    "_fit_eval", "_validate_indices", "_instantiate_strategy",
    "_copy_state", "_compute_auc",
    # constants
    "DEFAULT_TRACKED_METRICS", "DEFAULT_METRIC_KEYS", "DEFAULT_GAP_KEYS",
]
