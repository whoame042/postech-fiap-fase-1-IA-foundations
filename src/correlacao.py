"""Heatmap e pares |r| altos (T07). Correlação ≠ causalidade."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda import repo_root
from src.preprocess import load_and_clean

FIG_NAME = "reports/figures/02_correlacao.png"
LIMIAR = 0.90


def pearson_pairs(X: pd.DataFrame, limiar: float = LIMIAR) -> pd.Series:
    corr = X.corr(method="pearson")
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().abs().sort_values(ascending=False)
    return pairs[pairs >= limiar]


def save_heatmap(X: pd.DataFrame, path: Path | None = None) -> Path:
    dest = Path(path) if path else repo_root() / FIG_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    corr = X.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Correlação de Pearson entre as 30 features (sem o alvo)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    _, X, _ = load_and_clean()
    dest = save_heatmap(X)
    pairs = pearson_pairs(X)
    print("figura", dest)
    print("pares |r|>=0.90")
    print(pairs.head(15).to_string())
