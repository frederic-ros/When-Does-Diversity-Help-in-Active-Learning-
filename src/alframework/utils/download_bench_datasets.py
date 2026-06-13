# -*- coding: utf-8 -*-
"""
download_bench_datasets.py  —  acquisition datasets pour benchmark AL
======================================================================

Télécharge 5 datasets couvrant les zones non couvertes par newreel,
et les exporte dans le format bench_real.py :
  - fichier .txt, tabulation-séparé, label en DERNIÈRE colonne, sans en-tête

Fichiers produits pour chaque dataset :
  <nom>.txt          → données brutes (toujours produit)
  <nom>_pca95.txt    → données après ACP 95% var. (si --pca, datasets haute dim.)

Usage
-----
    # Télécharger tous les datasets (bruts uniquement)
    python download_bench_datasets.py --out ./newreel_extra

    # Avec fichiers PCA en plus (pour les datasets haute dimension)
    python download_bench_datasets.py --out ./newreel_extra --pca

    # Un seul dataset
    python download_bench_datasets.py --out ./newreel_extra --only mammography

Datasets
--------
  mammography   imbalance forte (2.3% positifs, 11 183 pts, 6 features)
  pendigits     10 classes, 10 992 pts, 16 features
  waveform      overlap fort, 3 classes, 5 000 pts, 40 features
  madelon       haute dimension, 500 features, 2 000 pts, binaire
  har           12 classes, 561 features, 10 299 pts  ← PCA utile pour LR

Dépendances
-----------
    pip install openml scikit-learn numpy pandas
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import io
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Registre des datasets ────────────────────────────────────────────────────
REGISTRY = {
    "mammography": {
        "openml_id": 310,
        "pca_candidate": False,
        "desc": "imbalance forte (2.3% positifs), 6 features, 11 183 pts",
    },
    "pendigits": {
        "openml_id": 32,
        "pca_candidate": False,
        "desc": "10 classes, 16 features, 10 992 pts",
    },
    "waveform": {
        "openml_id": 60,
        "pca_candidate": False,
        "desc": "overlap fort, 3 classes, 40 features, 5 000 pts",
    },
    "madelon": {
        "openml_id": 1485,
        "pca_candidate": False,   # PCA déconseillée : 480 features de bruit pur
        "desc": "500 features (480 bruit), 2 000 pts, binaire — PCA non recommandée",
    },
    "har": {
        "openml_id": 1478,
        "pca_candidate": True,
        "desc": "12 classes, 561 features, 10 299 pts — PCA recommandée pour LR",
    },
}

# URLs de fallback UCI / dépôts publics (si OpenML timeout)
UCI_FALLBACK = {
    "mammography": {
        "url": "https://archive.ics.uci.edu/static/public/161/mammography.zip",
        "type": "zip_csv",
        "filename_in_zip": "mammography.csv",
        "sep": ",",
        "label_col": "class",
        "label_map": {"'-1'": 0, "'1'": 1, "-1": 0, "1": 1},
    },
    "pendigits": {
        "url_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
        "url_test":  "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
        "type": "pendigits",
        "sep": ",",
    },
    "waveform": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform.data.Z",
        "type": "waveform",
    },
    "madelon": {
        "url_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data",
        "url_labels": "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels",
        "type": "madelon",
    },
    "har": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
        "type": "har_zip",
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ensure_openml() -> None:
    try:
        import openml  # noqa: F401
    except ImportError:
        print("[INSTALL] openml manquant → pip install openml ...")
        os.system(f"{sys.executable} -m pip install openml -q")

def _fetch_openml(openml_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Charge X, y depuis OpenML avec retry."""
    import openml
    for attempt in range(2):
        try:
            ds = openml.datasets.get_dataset(
                openml_id,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X, y, _, _ = ds.get_data(dataset_format="dataframe")
            X = pd.DataFrame(X)
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols):
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            X = X.fillna(X.median(numeric_only=True))
            X_arr = X.to_numpy(dtype=np.float64)
            from sklearn.preprocessing import LabelEncoder
            y_arr = LabelEncoder().fit_transform(y.astype(str))
            return X_arr, y_arr
        except Exception as e:
            if attempt == 0:
                print(f"    OpenML tentative 1 échouée ({type(e).__name__}), retry dans 5s ...")
                time.sleep(5)
            else:
                raise


def _download_url(url: str, timeout: int = 60) -> bytes:
    """Télécharge une URL et retourne les bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_uci(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback UCI direct quand OpenML est indisponible."""
    from sklearn.preprocessing import LabelEncoder
    cfg = UCI_FALLBACK.get(name)
    if cfg is None:
        raise ValueError(f"Pas de fallback UCI pour {name}")

    t = cfg["type"]
    print(f"    → Fallback UCI ({t}) ...")

    if t == "pendigits":
        raw_tr = _download_url(cfg["url_train"])
        raw_te = _download_url(cfg["url_test"])
        df_tr = pd.read_csv(io.BytesIO(raw_tr), header=None, sep=cfg["sep"])
        df_te = pd.read_csv(io.BytesIO(raw_te), header=None, sep=cfg["sep"])
        df = pd.concat([df_tr, df_te], ignore_index=True)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
        y = df.iloc[:, -1].to_numpy(dtype=np.int64)
        return X, y

    if t == "madelon":
        raw_X = _download_url(cfg["url_train"])
        raw_y = _download_url(cfg["url_labels"])
        X = pd.read_csv(io.BytesIO(raw_X), header=None, sep=" ").to_numpy(dtype=np.float64)
        y_raw = pd.read_csv(io.BytesIO(raw_y), header=None).values.ravel()
        y = LabelEncoder().fit_transform(y_raw)
        return X, y

    if t == "har_zip":
        raw = _download_url(cfg["url"], timeout=120)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            def _read(path):
                with z.open(path) as f:
                    return pd.read_csv(f, header=None, delim_whitespace=True)
            X_tr = _read("UCI HAR Dataset/train/X_train.txt").to_numpy(dtype=np.float64)
            X_te = _read("UCI HAR Dataset/test/X_test.txt").to_numpy(dtype=np.float64)
            y_tr = _read("UCI HAR Dataset/train/y_train.txt").values.ravel()
            y_te = _read("UCI HAR Dataset/test/y_test.txt").values.ravel()
        X = np.vstack([X_tr, X_te])
        y = LabelEncoder().fit_transform(np.concatenate([y_tr, y_te]))
        return X, y

    if t == "waveform":
        # waveform.data.Z = compress unix — fallback waveform-+noise version
        # Utiliser la version non-compressée waveform.data si disponible
        url_plain = cfg["url"].replace(".data.Z", ".data")
        try:
            raw = _download_url(url_plain)
        except Exception:
            raw = _download_url(cfg["url"].replace(".data.Z", "+noise.data"))
        df = pd.read_csv(io.BytesIO(raw), header=None)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
        y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
        return X, y

    if t == "zip_csv":
        raw = _download_url(cfg["url"])
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            fname = cfg.get("filename_in_zip") or z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f, sep=cfg.get("sep", ","))
        label_col = cfg.get("label_col")
        if label_col and label_col in df.columns:
            y_raw = df[label_col].astype(str)
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float64)
        else:
            X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
            y_raw = df.iloc[:, -1].astype(str)
        lmap = cfg.get("label_map", {})
        if lmap:
            y = np.array([lmap.get(str(v).strip(), v) for v in y_raw], dtype=np.int64)
        else:
            y = LabelEncoder().fit_transform(y_raw)
        return X, y

    raise ValueError(f"Type UCI inconnu : {t}")

def _apply_pca(
    X: np.ndarray,
    variance: float = 0.95,
    min_components: int = 10,
) -> Tuple[np.ndarray, int]:
    """Retourne X réduit par PCA + nombre de composantes retenues."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X_sc = StandardScaler().fit_transform(X)
    n_max = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_max, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_keep = max(min_components, int(np.searchsorted(cum_var, variance) + 1))
    n_keep = min(n_keep, X_pca.shape[1])
    return X_pca[:, :n_keep], n_keep

def _save_txt(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """
    Sauvegarde dans le format bench_real.py :
      - tabulation comme séparateur
      - label (entier) en DERNIÈRE colonne
      - pas d'en-tête
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for xi, yi in zip(X, y):
            feats = "\t".join(f"{v:.10g}" for v in xi)
            f.write(f"{feats}\t{int(yi)}\n")

def _describe(name: str, X: np.ndarray, y: np.ndarray) -> None:
    classes, counts = np.unique(y, return_counts=True)
    imb = max(counts) / max(min(counts), 1)
    print(f"    n={len(y):>8}  features={X.shape[1]:>4}  classes={len(classes):>3}")
    print(f"    imbalance ratio={imb:.1f}×  "
          f"min_class={min(counts)}  max_class={max(counts)}")

# ─── Téléchargement d'un dataset ──────────────────────────────────────────────
def download_one(
    name: str,
    out_dir: Path,
    apply_pca: bool = False,
    pca_variance: float = 0.95,
    force: bool = False,
) -> dict:
    cfg = REGISTRY.get(name)
    if cfg is None:
        print(f"  [SKIP] '{name}' inconnu. Choix : {list(REGISTRY)}")
        return {"raw": None, "pca": None}

    print(f"\n{'─'*65}")
    print(f"  Dataset : {name}")
    print(f"  Info    : {cfg['desc']}")
    print(f"  OpenML  : {cfg['openml_id']}")

    raw_path = out_dir / f"{name}.txt"
    pca_path = out_dir / f"{name}_pca{int(pca_variance*100)}.txt"

    # Vérifier si déjà présent
    raw_exists = raw_path.exists() and not force
    pca_exists = pca_path.exists() and not force

    if raw_exists and (not apply_pca or pca_exists or not cfg["pca_candidate"]):
        print(f"  ✓ Déjà présent : {raw_path.name}"
              + (f" + {pca_path.name}" if pca_exists else ""))
        print(f"    (utiliser --force pour re-télécharger)")
        return {"raw": raw_path, "pca": pca_path if pca_exists else None}

    # Téléchargement
    print(f"  Téléchargement en cours ...")
    X, y = None, None
    # Tentative 1 : OpenML
    try:
        _ensure_openml()
        X, y = _fetch_openml(cfg["openml_id"])
        print(f"    ✓ OpenML OK")
    except Exception as e:
        print(f"    OpenML indisponible ({type(e).__name__}) → fallback UCI ...")
        # Tentative 2 : UCI direct
        try:
            X, y = _fetch_uci(name)
            print(f"    ✓ UCI fallback OK")
        except Exception as e2:
            print(f"  [ERREUR] OpenML + UCI échoués : {e2}")
            return {"raw": None, "pca": None}

    _describe(name, X, y)

    # ── Fichier brut (toujours produit) ───────────────────────────────────────
    if not raw_exists:
        _save_txt(X, y, raw_path)
        sz = raw_path.stat().st_size // 1024
        print(f"  ✓ Brut sauvegardé  → {raw_path.name}  ({sz} Ko)")
    else:
        print(f"  ✓ Brut déjà présent → {raw_path.name}")

    # ── Fichier PCA (optionnel) ────────────────────────────────────────────────
    result = {"raw": raw_path, "pca": None}
    if apply_pca:
        if not cfg["pca_candidate"]:
            print(f"  ⚠  PCA ignorée pour {name} :")
            print(f"     → {cfg['desc']}")
            print(f"     → Le fichier brut est le bon à utiliser.")
        elif pca_exists:
            print(f"  ✓ PCA déjà présent → {pca_path.name}")
            result["pca"] = pca_path
        else:
            print(f"  ACP en cours ({X.shape[1]} features → {pca_variance:.0%} variance) ...")
            X_pca, n_keep = _apply_pca(X, variance=pca_variance)
            _save_txt(X_pca, y, pca_path)
            sz = pca_path.stat().st_size // 1024
            print(f"  ✓ PCA sauvegardée  → {pca_path.name}  "
                  f"({X.shape[1]} → {n_keep} composantes, {sz} Ko)")
            result["pca"] = pca_path

    return result

# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge et prépare des datasets réels pour bench_real.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python download_bench_datasets.py --out ./newreel_extra
  python download_bench_datasets.py --out ./newreel_extra --pca
  python download_bench_datasets.py --out ./newreel_extra --only mammography pendigits
  python download_bench_datasets.py --out ./newreel_extra --pca --pca-variance 0.99 --force
        """,
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./newreel_extra"),
        help="Dossier de sortie (défaut : ./newreel_extra)",
    )
    parser.add_argument(
        "--only", nargs="+", choices=list(REGISTRY), default=None,
        help="Télécharger seulement ces datasets",
    )
    parser.add_argument(
        "--pca", action="store_true",
        help="Produire aussi un fichier *_pcaXX.txt pour les datasets haute dimension",
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="Variance cible pour l'ACP (défaut : 0.95)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-télécharger même si les fichiers existent déjà",
    )
    args = parser.parse_args()

    to_download: List[str] = args.only or list(REGISTRY)
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Benchmark AL — Acquisition datasets supplémentaires")
    print("=" * 65)
    print(f"  Dossier de sortie : {args.out.resolve()}")
    print(f"  Datasets          : {to_download}")
    print(f"  Fichiers PCA      : {'OUI (variance=' + str(args.pca_variance) + ')' if args.pca else 'NON'}")
    print(f"  Format            : .txt, tabulation, label en dernière colonne")

    results = {}
    for name in to_download:
        results[name] = download_one(
            name=name,
            out_dir=args.out,
            apply_pca=args.pca,
            pca_variance=args.pca_variance,
            force=args.force,
        )

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Résumé")
    print(f"  {'Dataset':<14}  {'Brut':>12}  {'PCA':>16}  Statut")
    print("  " + "─" * 58)
    for name, r in results.items():
        raw_s  = r["raw"].name  if r["raw"]  else "—"
        pca_s  = r["pca"].name  if r["pca"]  else ("—" if args.pca else "non demandé")
        status = "✓" if r["raw"] else "✗ ERREUR"
        raw_sz = f"({r['raw'].stat().st_size//1024}Ko)" if r["raw"] else ""
        print(f"  {name:<14}  {raw_s+' '+raw_sz:>18}  {pca_s:>18}  {status}")

    print(f"""
  Pour utiliser dans bench_real.py :
    bench_root = Path("{args.out.resolve()}")

  Note sur la PCA :
    - mammography, pendigits, waveform : PCA inutile (≤ 40 features)
    - madelon : PCA déconseillée (480 features de bruit = le test lui-même)
    - har : PCA recommandée pour LR (561 features corrélées → ~80 composantes)
            Utiliser har.txt pour RF, har_pca95.txt pour LR
""")


if __name__ == "__main__":
    main()
