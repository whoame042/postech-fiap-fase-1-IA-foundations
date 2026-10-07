"""Split único (T08) e treino T09–T11.

O teste não é usado até a T12. Métricas daqui são só de validação.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.eda import repo_root
from src.preprocess import build_preprocess_pipeline, load_and_clean

SEED = 42
TEST_SIZE = 0.15
# 0.15 / 0.85 ≈ 0.1765 → ~70/15/15 do N original
VAL_SIZE_FROM_TRAINVAL = 0.1765
SPLIT_PATH = "models/split.joblib"


def make_split(X, y, seed: int = SEED):
    """Retorna X_train, X_val, X_test, y_train, y_val, y_test (estratificado)."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_FROM_TRAINVAL,
        stratify=y_trainval,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _indices(frame) -> list:
    return list(frame.index)


def assert_disjoint(X_train, X_val, X_test) -> None:
    sets = [set(_indices(X_train)), set(_indices(X_val)), set(_indices(X_test))]
    if set.intersection(*sets[:2]) or set.intersection(sets[0], sets[2]) or set.intersection(sets[1], sets[2]):
        raise ValueError("split com interseção de índices")
    n = len(sets[0]) + len(sets[1]) + len(sets[2])
    if n != len(X_train) + len(X_val) + len(X_test):
        raise ValueError("tamanhos inconsistentes")


def save_split(X_train, X_val, X_test, seed: int = SEED, path: Path | None = None) -> Path:
    dest = Path(path) if path else repo_root() / SPLIT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "test_size": TEST_SIZE,
        "val_size_from_trainval": VAL_SIZE_FROM_TRAINVAL,
        "proporcao": "70/15/15",
        "train_idx": _indices(X_train),
        "val_idx": _indices(X_val),
        "test_idx": _indices(X_test),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    joblib.dump(payload, dest)
    return dest


SCORING = "recall"


def load_locked_split():
    """Reconstrói o split da T08. Não sorteia de novo."""
    _, X, y = load_and_clean()
    split = joblib.load(repo_root() / SPLIT_PATH)
    X_train, y_train = X.loc[split["train_idx"]], y.loc[split["train_idx"]]
    X_val, y_val = X.loc[split["val_idx"]], y.loc[split["val_idx"]]
    X_test, y_test = X.loc[split["test_idx"]], y.loc[split["test_idx"]]
    assert_disjoint(X_train, X_val, X_test)
    if list(X_train.columns) != list(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise ValueError("features divergem entre splits")
    return X_train, X_val, X_test, y_train, y_val, y_test, split


def val_metrics(y_true, y_pred) -> dict:
    return {
        "conjunto": "validacao",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _full_pipe(X_train, clf) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocess_pipeline(X_train)),
            ("clf", clf),
        ]
    )


def _fit_grid(pipe, param_grid, X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring=SCORING,
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def _persist(name: str, estimator, X_val, y_val, grid: GridSearchCV) -> dict:
    root = repo_root()
    y_pred = estimator.predict(X_val)
    proba = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_val)[:, 1]
    metrics = val_metrics(y_val, y_pred)
    metrics["modelo"] = name
    metrics["scoring_grid"] = SCORING
    metrics["best_params"] = {
        k: (None if v is None else (v.item() if hasattr(v, "item") else v))
        for k, v in grid.best_params_.items()
    }
    metrics["best_cv_recall"] = float(grid.best_score_)
    metrics["n_features"] = int(X_val.shape[1])
    metrics_path = root / "reports" / f"metrics_val_{name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pred = pd.DataFrame({"y_true": y_val.to_numpy(), "y_pred": y_pred}, index=X_val.index)
    if proba is not None:
        pred["y_proba_maligno"] = proba
    pred_path = root / "reports" / f"pred_val_{name}.csv"
    pred.to_csv(pred_path, index_label="row_idx")
    model_path = root / "models" / f"{name}.joblib"
    joblib.dump(estimator, model_path)
    return {"metrics": metrics, "model": model_path, "pred": pred_path, "cv": grid.cv_results_}


def train_logreg(X_train, y_train, X_val, y_val) -> dict:
    pipe = _full_pipe(
        X_train,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    grid = _fit_grid(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, X_train, y_train)
    return _persist("logreg", grid.best_estimator_, X_val, y_val, grid)


def train_rf(X_train, y_train, X_val, y_val) -> dict:
    pipe = _full_pipe(
        X_train,
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    )
    grid = _fit_grid(
        pipe,
        {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10]},
        X_train,
        y_train,
    )
    return _persist("rf", grid.best_estimator_, X_val, y_val, grid)


def train_knn(X_train, y_train, X_val, y_val) -> dict:
    pipe = _full_pipe(X_train, KNeighborsClassifier())
    grid = _fit_grid(
        pipe,
        {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
        X_train,
        y_train,
    )
    return _persist("knn", grid.best_estimator_, X_val, y_val, grid)


def train():
    X_train, X_val, X_test, y_train, y_val, y_test, split = load_locked_split()
    del X_test, y_test  # lacre: não calcular métrica de teste aqui
    print("split_seed", split["seed"], "n_train", len(X_train), "n_val", len(X_val))
    print("features", list(X_train.columns)[:3], "...", len(X_train.columns))
    for fn in (train_logreg, train_rf, train_knn):
        out = fn(X_train, y_train, X_val, y_val)
        m = out["metrics"]
        print(m["modelo"], m["best_params"], "val_recall", round(m["recall"], 4), "val_f1", round(m["f1"], 4))


if __name__ == "__main__":
    train()
