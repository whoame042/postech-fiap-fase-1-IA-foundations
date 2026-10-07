"""EDA do Wisconsin Diagnostic (T03–T04). Não treina modelo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "data.csv"
FIG_DIR = ROOT / "reports" / "figures"

# 1 = maligno (classe positiva de triagem). Não inverter.
DIAGNOSIS_MAP = {"M": 1, "B": 0}
DIAGNOSIS_LABEL = {1: "maligno", 0: "benigno"}

FEATURES_BOX = [
    "concave points_worst",
    "perimeter_worst",
    "radius_worst",
    "area_worst",
    "concave points_mean",
    "radius_mean",
]
FEATURES_HIST = ["concave points_worst", "radius_mean"]
FEATURES_PAIR = [
    "radius_mean",
    "concave points_mean",
    "area_worst",
    "texture_mean",
]


def repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data" / "data.csv").exists():
        return cwd
    if (cwd.parent / "data" / "data.csv").exists():
        return cwd.parent
    return ROOT


def load_frame(csv_path: Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else repo_root() / "data" / "data.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    if "diagnosis" not in df.columns:
        raise ValueError("coluna diagnosis ausente")
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X sem id e sem alvo. y: 1 = maligno, 0 = benigno."""
    drop = [c for c in ("id", "diagnosis") if c in df.columns]
    X = df.drop(columns=drop)
    y = df["diagnosis"].map(DIAGNOSIS_MAP)
    if y.isna().any():
        raise ValueError("diagnosis com valor fora de M/B")
    return X, y.astype(int)


def summarize(df: pd.DataFrame) -> dict:
    X, y = split_xy(df)
    return {
        "shape_bruto": df.shape,
        "shape_X": X.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nunique": df.nunique().to_dict(),
        "features": list(X.columns),
        "id_em_X": "id" in X.columns,
        "alvo": y.value_counts().rename(DIAGNOSIS_LABEL).to_dict(),
        "alvo_prop": y.value_counts(normalize=True).rename(DIAGNOSIS_LABEL).to_dict(),
    }


def save_figures(df: pd.DataFrame, fig_dir: Path | None = None) -> list[Path]:
    fig_dir = Path(fig_dir) if fig_dir else repo_root() / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    X, y = split_xy(df)
    plot_df = X.copy()
    plot_df["diagnostico"] = y.map(DIAGNOSIS_LABEL)

    sns.set_theme(style="whitegrid", context="talk")
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["benigno", "maligno"]
    counts = plot_df["diagnostico"].value_counts().reindex(order)
    colors = ["#4C78A8", "#E45756"]
    ax.bar(order, counts.values, color=colors)
    ax.set_title("Contagem do alvo (diagnosis)")
    ax.set_xlabel("Diagnóstico do exame (rótulo da base)")
    ax.set_ylabel("Número de exames")
    for i, n in enumerate(counts.values):
        ax.text(i, n + 4, str(int(n)), ha="center")
    fig.tight_layout()
    p = fig_dir / "01_alvo_contagem.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), FEATURES_BOX):
        sns.boxplot(
            data=plot_df,
            x="diagnostico",
            y=col,
            hue="diagnostico",
            order=order,
            palette={"benigno": "#4C78A8", "maligno": "#E45756"},
            legend=False,
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("Diagnóstico")
        ax.set_ylabel(col)
    fig.suptitle("Box-plots: features com maior separação visual entre classes")
    fig.tight_layout()
    p = fig_dir / "01_boxplots_por_diagnostico.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, FEATURES_HIST):
        sns.histplot(
            data=plot_df,
            x=col,
            hue="diagnostico",
            hue_order=order,
            palette={"benigno": "#4C78A8", "maligno": "#E45756"},
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Densidade de {col} por classe")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")
    fig.tight_layout()
    p = fig_dir / "01_histogramas_por_classe.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    pair = plot_df[FEATURES_PAIR + ["diagnostico"]]
    g = sns.pairplot(
        pair,
        hue="diagnostico",
        hue_order=order,
        palette={"benigno": "#4C78A8", "maligno": "#E45756"},
        corner=True,
        diag_kind="kde",
    )
    g.fig.suptitle("Subset (4 features): overlap entre maligno e benigno", y=1.02)
    p = fig_dir / "01_pairplot_subset.png"
    g.savefig(p, dpi=120)
    plt.close(g.fig)
    saved.append(p)

    return saved


if __name__ == "__main__":
    frame = load_frame()
    info = summarize(frame)
    print("shape_bruto", info["shape_bruto"])
    print("shape_X", info["shape_X"], "n_features", len(info["features"]))
    print("id_em_X", info["id_em_X"])
    print("alvo", info["alvo"], info["alvo_prop"])
    print("describe\n", split_xy(frame)[0].describe().T.head())
    paths = save_figures(frame)
    for path in paths:
        print("figura", path)
