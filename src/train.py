"""Split único e, depois, treino (T08 / T09–T11).

O teste não é usado até a T12. Não treinar neste módulo ainda.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.eda import repo_root
from src.preprocess import load_and_clean

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


def train():
    raise NotImplementedError("T09–T11: treinar via pipeline de T06 só em X_train")


if __name__ == "__main__":
    _, X, y = load_and_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y)
    assert_disjoint(X_train, X_val, X_test)
    dest = save_split(X_train, X_val, X_test)
    print("n", len(X), "train", len(X_train), "val", len(X_val), "test", len(X_test))
    print("soma", len(X_train) + len(X_val) + len(X_test))
    print("prop_maligno_train", float(y_train.mean()))
    print("prop_maligno_val", float(y_val.mean()))
    print("prop_maligno_test", float(y_test.mean()))
    print("split", dest)
    # y_test existe no split; não reportar métrica de teste (T12).
