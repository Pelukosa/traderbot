#!/usr/bin/env python3
"""
Entrenamiento del modelo XGBoost con optimización de hiperparámetros (Optuna).

Lee BTCUSD_1h_Binance.csv, calcula 31 features técnicas, entrena y guarda
el mejor modelo en ml_strategy/models/bitcoin_xgb_model.json.

Uso:
    python -m ml_strategy.train_model                           # defaults + Optuna
    python -m ml_strategy.train_model --no-optuna               # sin optimizar
    python -m ml_strategy.train_model --target-pct 2.0 --horizon 6
    python -m ml_strategy.train_model --trials 100              # 100 intentos Optuna
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ── Añadir la raíz del proyecto al path para imports ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_strategy.features import compute_features, FEATURE_COLUMNS  # type: ignore

# ── Parámetros por defecto ──
DEFAULT_DATA = _PROJECT_ROOT / "BTCUSD_1h_Binance.csv"
DEFAULT_TARGET_PCT = 2.0
DEFAULT_HORIZON = 6
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_NAME = "bitcoin_xgb_model.json"
DEFAULT_OPTUNA_TRIALS = 50


def _create_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    target_pct: float,
    horizon: int,
) -> np.ndarray:
    """
    Etiquetas con tri-barrier method simplificado.

    Para cada vela i, mira hacia adelante `horizon` velas:
      1 → el precio toca +target_pct% ANTES de tocar -target_pct%
      0 → toca -target_pct% primero o no toca ninguno

    Más realista que un simple "sube X% en N velas": captura
    la dirección del primer movimiento significativo, no solo el cierre final.
    """
    n = len(closes)
    labels = np.empty(n, dtype=np.float64)
    labels[:] = np.nan

    for i in range(n - horizon):
        entry = closes[i]
        if entry <= 0:
            labels[i] = 0.0
            continue

        upper_target = entry * (1.0 + target_pct / 100.0)
        lower_target = entry * (1.0 - target_pct / 100.0)

        hit_upper = False
        hit_lower = False
        for j in range(1, horizon + 1):
            h = highs[i + j]
            l = lows[i + j]
            if h >= upper_target:
                hit_upper = True
                break
            if l <= lower_target:
                hit_lower = True
                break

        labels[i] = 1.0 if hit_upper else 0.0

    return labels


def load_and_prepare(
    csv_path: Path,
    target_pct: float,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga el CSV, calcula features + labels, dropea NaNs.

    Retorna (X, y) donde X es DataFrame de features e y es array de labels.
    """
    print(f"[train] Cargando datos desde {csv_path} …")
    df = pd.read_csv(csv_path)

    # ── Detectar formato de columnas y normalizar ──
    # Formato 1: timestamp, open, high, low, close, volume
    # Formato 2: Open time, Open, High, Low, Close, Volume (Binance)
    # Formato 3: nombres en español (apertura, maximo, minimo, cierre, volumen)
    col_map: dict[str, str] = {}
    time_col: str | None = None

    for col in df.columns:
        key = col.strip().lower().replace(" ", "")
        if key in ("timestamp", "opentime"):
            time_col = col
        elif key in ("open", "apertura"):
            col_map[col] = "open"
        elif key in ("high", "maximo", "máximo"):
            col_map[col] = "high"
        elif key in ("low", "minimo", "mínimo"):
            col_map[col] = "low"
        elif key in ("close", "cierre"):
            col_map[col] = "close"
        elif key in ("volume", "volumen"):
            col_map[col] = "volume"

    df.rename(columns=col_map, inplace=True)

    # Usar columna de tiempo como índice
    if time_col is None:
        # Fallback: buscar cualquier columna que parezca datetime
        for col in df.columns:
            if "time" in col.lower() or "fecha" in col.lower() or "timestamp" in col.lower():
                time_col = col
                break
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    elif df.index.name is None or df.index.dtype.kind not in ("M", "i"):
        print("[train] WARNING — No se detectó columna de tiempo. Usando índice numérico.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"[train] ERROR — Faltan columnas: {missing}")
        sys.exit(1)

    # Ordenar por índice temporal
    df.sort_index(inplace=True)

    print(f"[train] {len(df):,} velas cargadas — {df.index[0]} → {df.index[-1]}")

    # ── Features ──
    print("[train] Calculando features …")
    feats = compute_features(df)

    # ── Labels (tri-barrier) ──
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    labels = _create_labels(closes, highs, lows, target_pct, horizon)

    # ── Unir y dropear NaNs ──
    feats["target"] = labels
    clean = feats.dropna()

    X = clean[FEATURE_COLUMNS]
    y = clean["target"].values.astype(int)

    print(f"[train] Dataset limpio: {len(X):,} filas × {len(FEATURE_COLUMNS)} features")
    print(f"[train] Balance de clases — 0: {(y == 0).sum():,}  |  1: {(y == 1).sum():,}  "
          f"(ratio 1 = {y.mean():.2%})")

    return X, y


# ═══════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════

def walk_forward_evaluate(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    purge: int = 12,
) -> dict[str, float]:
    """Walk-forward con purge para evitar data leakage."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    metrics: dict[str, list[float]] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for k in range(n_splits):
        train_end = (k + 1) * fold_size
        test_start = train_end + purge
        test_end = min(test_start + fold_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        X_tr, X_te = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_tr, y_te = y[:train_end], y[test_start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_te)

        metrics["accuracy"].append(accuracy_score(y_te, y_pred))
        metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))

        print(f"  WF fold {k + 1}: acc={metrics['accuracy'][-1]:.3f}  "
              f"prec={metrics['precision'][-1]:.3f}  "
              f"rec={metrics['recall'][-1]:.3f}  "
              f"f1={metrics['f1'][-1]:.3f}  "
              f"(train={len(X_tr):,}, test={len(y_te):,})")

    if not metrics["f1"]:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════
# Optuna
# ═══════════════════════════════════════════════════════════════════

def _optuna_objective(
    trial,
    X: pd.DataFrame,
    y: np.ndarray,
    scale_pos_weight: float,
    cv_splits: int,
) -> float:
    """Objetivo de Optuna: maximizar F1 en TimeSeriesSplit."""
    from xgboost import XGBClassifier  # type: ignore

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "early_stopping_rounds": 20,
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    f1_scores: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def optimize_hyperparams(
    X: pd.DataFrame, y: np.ndarray, n_trials: int = 50, cv_splits: int = 3
) -> dict[str, Any]:
    """Ejecuta Optuna y devuelve los mejores hiperparámetros."""
    import optuna  # type: ignore

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    sw = n_neg / max(n_pos, 1)

    print(f"\n[train] Optuna — {n_trials} trials, scale_pos_weight={sw:.2f}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective(trial, X, y, sw, cv_splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[train] Mejor trial #{study.best_trial.number}: F1={study.best_value:.4f}")
    return {**best, "scale_pos_weight": sw, "random_state": 42, "n_jobs": -1,
            "tree_method": "hist", "eval_metric": "logloss"}


# ═══════════════════════════════════════════════════════════════════
# Entrenamiento final
# ═══════════════════════════════════════════════════════════════════

def train(
    X: pd.DataFrame, y: np.ndarray, model_path: Path,
    use_optuna: bool = True, n_trials: int = DEFAULT_OPTUNA_TRIALS, cv_splits: int = 5,
) -> None:
    from xgboost import XGBClassifier  # type: ignore

    if use_optuna:
        params = optimize_hyperparams(X, y, n_trials=n_trials, cv_splits=min(cv_splits, 3))
    else:
        sw = (y == 0).sum() / max((y == 1).sum(), 1)
        params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "gamma": 0.1, "reg_alpha": 0.5, "reg_lambda": 1.0,
            "scale_pos_weight": sw, "random_state": 42, "n_jobs": -1,
            "tree_method": "hist", "eval_metric": "logloss",
        }

    # Walk-forward
    print(f"\n[train] Walk-forward validation ({cv_splits} folds, purge=12) …")
    wf_model = XGBClassifier(**params)
    wf_metrics = walk_forward_evaluate(wf_model, X, y, n_splits=cv_splits, purge=12)
    print(f"[train] WF promedio:  acc={wf_metrics['accuracy']:.4f}  "
          f"prec={wf_metrics['precision']:.4f}  rec={wf_metrics['recall']:.4f}  "
          f"f1={wf_metrics['f1']:.4f}")

    # Entrenamiento final
    print("\n[train] Entrenando modelo final …")
    params.pop("early_stopping_rounds", None)
    final_model = XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    # Guardar
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))

    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "params": {k: v for k, v in params.items()
                   if k not in ("n_jobs", "tree_method", "eval_metric")},
        "wf_metrics": wf_metrics,
        "class_balance": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
    }
    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n[train] ✅ Modelo guardado en {model_path}")
    print(f"[train] ✅ Metadatos guardados en {meta_path}")

    # Feature importance
    importance = final_model.feature_importances_
    print("\n[train] Top 10 features por importancia:")
    for idx in np.argsort(importance)[::-1][:10]:
        print(f"  {FEATURE_COLUMNS[idx]:>25s}: {importance[idx]:.4f}")

    low_imp = [(FEATURE_COLUMNS[i], importance[i]) for i in range(len(importance))
               if importance[i] < 0.01]
    if low_imp:
        print(f"\n[train] ⚠️  {len(low_imp)} features irrelevantes (<1% importancia):")
        for name, imp in sorted(low_imp, key=lambda x: x[1]):
            print(f"  {name:>25s}: {imp:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena XGBoost para señales de trading (con Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target-pct", type=float, default=DEFAULT_TARGET_PCT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--no-optuna", action="store_true",
                        help="Desactiva Optuna (entrenamiento rápido)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[train] ERROR — No se encuentra {args.data}")
        sys.exit(1)

    print("=" * 65)
    print("  XGBoost ML Strategy — Entrenamiento Avanzado")
    print(f"  target_pct = {args.target_pct}%  |  horizon = {args.horizon}h  |  "
          f"cv = {args.cv} folds")
    print(f"  Optuna = {'OFF' if args.no_optuna else f'ON ({args.trials} trials)'}")
    print("  Labels = tri-barrier (toca +/-target% primero)")
    print("=" * 65)

    X, y = load_and_prepare(args.data, args.target_pct, args.horizon)
    train(X, y, args.output,
          use_optuna=not args.no_optuna,
          n_trials=args.trials,
          cv_splits=args.cv)

    print("\n[train] Listo. Para predecir:")
    print("  from ml_strategy import fetch_and_predict")
    print("  signal = await fetch_and_predict()")


if __name__ == "__main__":
    main()
