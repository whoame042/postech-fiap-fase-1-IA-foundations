"""T12–T14: vencedor na validação, teste uma vez, importance e SHAP."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.eda import repo_root
from src.train import load_locked_split

MODELS = ("logreg", "rf", "knn")
LABELS = ("benigno", "maligno")


def _root() -> Path:
    return repo_root()


def load_val_table() -> pd.DataFrame:
    rows = []
    for name in MODELS:
        rows.append(json.loads((_root() / "reports" / f"metrics_val_{name}.json").read_text()))
    return pd.DataFrame(rows)


def choose_winner(val: pd.DataFrame) -> str:
    """Recall manda; F1 desempata. Só números de validação."""
    ranked = val.sort_values(["recall", "f1"], ascending=False)
    return str(ranked.iloc[0]["modelo"])


def test_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "conjunto": "teste",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def save_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Falso positivo (1 − especificidade)")
    ax.set_ylabel("Recall (sensibilidade)")
    ax.set_title("ROC no teste (uma vez)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_names(pipe) -> np.ndarray:
    names = pipe.named_steps["prep"].get_feature_names_out()
    return np.array([n.split("__", 1)[-1] for n in names])


def save_importance(pipe, path: Path) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    names = feature_names(pipe)
    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
        kind = "coeficiente (escala do StandardScaler)"
    else:
        values = clf.feature_importances_
        kind = "feature_importances_"
    table = pd.DataFrame({"feature": names, "valor": values, "abs": np.abs(values)})
    table = table.sort_values("abs", ascending=False)
    top = table.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["valor"], color="#4C78A8")
    ax.set_xlabel(kind)
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 importâncias do vencedor (não é causalidade)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return table


def save_shap(pipe, X_train, X_test, y_test, y_pred, fig_dir: Path) -> dict:
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X_train_t = prep.transform(X_train)
    X_test_t = prep.transform(X_test)
    names = feature_names(pipe)
    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_train_t)
    elif hasattr(clf, "estimators_"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_train_t, 50, random_state=42))
    sv = explainer(X_test_t)
    if hasattr(sv, "values") and getattr(sv.values, "ndim", 1) == 3:
        sv = sv[:, :, 1]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, features=X_test_t, feature_names=names, show=False)
    plt.tight_layout()
    summary_path = fig_dir / "05_shap_summary.png"
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()

    y_test_np = np.asarray(y_test)
    fn = np.where((y_test_np == 1) & (y_pred == 0))[0]
    tp = np.where((y_test_np == 1) & (y_pred == 1))[0]
    idx = int(fn[0] if len(fn) else tp[0])
    row_idx = int(X_test.index[idx])
    plt.figure()
    shap.plots.waterfall(sv[idx], show=False)
    water_path = fig_dir / "05_shap_waterfall_caso.png"
    plt.savefig(water_path, dpi=120, bbox_inches="tight")
    plt.close()
    return {
        "row_idx": row_idx,
        "posicao_no_teste": idx,
        "y_real": int(y_test_np[idx]),
        "y_pred": int(y_pred[idx]),
        "summary": str(summary_path),
        "waterfall": str(water_path),
    }


def evaluate() -> dict:
    root = _root()
    fig = root / "reports" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    val = load_val_table()
    winner = choose_winner(val)
    X_train, X_val, X_test, y_train, y_val, y_test, split = load_locked_split()
    del X_val, y_val, y_train

    rows = []
    curves = []
    preds = {}
    for name in MODELS:
        pipe = joblib.load(root / "models" / f"{name}.joblib")
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        preds[name] = y_pred
        m = test_metrics(y_test, y_pred, y_proba)
        m["modelo"] = name
        m["vencedor_val"] = name == winner
        rows.append(m)
        save_confusion(
            y_test,
            y_pred,
            f"Matriz de confusão no teste — {name}",
            fig / f"03_cm_{name}.png",
        )
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        curves.append((name, fpr, tpr, m["roc_auc"]))
    save_roc(curves, fig / "03_roc.png")
    csv_path = root / "reports" / "metrics_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    winner_pipe = joblib.load(root / "models" / f"{winner}.joblib")
    imp = save_importance(winner_pipe, fig / "04_importance.png")
    shap_meta = save_shap(winner_pipe, X_train, X_test, y_test, preds[winner], fig)
    return {
        "vencedor": winner,
        "split_seed": split["seed"],
        "n_test": len(X_test),
        "metrics_test": rows,
        "importance_top": imp.head(5).to_dict(orient="records"),
        "shap_caso": shap_meta,
    }


if __name__ == "__main__":
    out = evaluate()
    print("vencedor_val", out["vencedor"])
    for row in out["metrics_test"]:
        print(row["modelo"], {k: row[k] for k in ("accuracy", "recall", "f1", "roc_auc", "fn", "fp")})
    print("shap_caso", out["shap_caso"])
