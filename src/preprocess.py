"""Limpeza sem estatística global + factory do pipeline (T05–T06).

Scaler/imputer só entram no ColumnTransformer. Fit ocorre depois do split (T09).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda import DIAGNOSIS_MAP, repo_root

UNFITTED_PATH = "models/preprocess_unfitted.joblib"


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else repo_root() / "data" / "data.csv"
    df = pd.read_csv(path)
    return df


def load_and_clean(csv_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Drop de coluna inútil / Unnamed / duplicata. Sem scaler e sem fillna global.

    Retorna (df_limpo com diagnosis 0/1, X, y). y: 1 = maligno.
    """
    df = load_raw(csv_path)
    shape_antes = df.shape
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().map(DIAGNOSIS_MAP)
    if df["diagnosis"].isna().any():
        raise ValueError("diagnosis com valor fora de M/B")
    df["diagnosis"] = df["diagnosis"].astype(int)
    y = df["diagnosis"]
    X = df.drop(columns=["diagnosis"])
    meta = {
        "shape_antes": shape_antes,
        "shape_depois": df.shape,
        "shape_X": X.shape,
        "duplicatas_removidas": n_dup,
        "na_por_coluna": X.isna().sum().to_dict(),
        "negativos": bool((X.select_dtypes("number") < 0).any().any()),
    }
    df.attrs["limpeza"] = meta
    return df, X, y


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return list(X.select_dtypes(include="number").columns)


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in numeric_columns(X)]


def build_preprocess(num_cols: list[str], cat_cols: list[str] | None = None) -> ColumnTransformer:
    """Factory. Não fita. Numéricas: mediana + StandardScaler. Categóricas: moda + OneHot."""
    cat_cols = list(cat_cols or [])
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = [("num", numeric, list(num_cols))]
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))
    return ColumnTransformer(transformers)


def build_preprocess_pipeline(X: pd.DataFrame | None = None) -> ColumnTransformer:
    if X is None:
        _, X, _ = load_and_clean()
    return build_preprocess(numeric_columns(X), categorical_columns(X))


def dump_unfitted(path: Path | None = None) -> Path:
    _, X, _ = load_and_clean()
    ct = build_preprocess_pipeline(X)
    dest = Path(path) if path else repo_root() / UNFITTED_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ct, dest)
    return dest


if __name__ == "__main__":
    df, X, y = load_and_clean()
    meta = df.attrs["limpeza"]
    print("limpeza", meta)
    print("y dtype", y.dtype, "unicos", sorted(y.unique()))
    dest = dump_unfitted()
    loaded = joblib.load(dest)
    print("unfitted", dest, type(loaded).__name__)
