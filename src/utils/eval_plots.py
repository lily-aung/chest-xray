from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, \
    precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_png: Path, normalize: bool = False, title: str = "Confusion Matrix"):
    _ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(cm, cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
    ax.set_title(title, fontsize=16, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    if class_names:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text( j, i,  f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=11, 
                    color="white" if cm[i, j] > threshold else "black", fontweight="bold" )
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_reliability(probs: np.ndarray, y_true: np.ndarray, out_png: Path, n_bins: int = 10, title: str = "Reliability"):
    if probs is None or probs.size == 0 or len(y_true) == 0:
        return
    _ensure_parent(out_png)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin_center": (lo + hi) / 2, "confidence": conf[mask].mean(), "accuracy": acc[mask].mean(), "count": int(mask.sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 6))
    sns.lineplot(data=df, x="confidence", y="accuracy", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roc_ovr(probs: np.ndarray, y_true: np.ndarray, class_names: list[str], out_png: Path, title: str = "ROC Curves (OvR)"):
    if probs is None or probs.size == 0:
        return
    _ensure_parent(out_png)
    n_classes = probs.shape[1]
    y = np.eye(n_classes)[y_true]
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y[:, c], probs[:, c])
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr_ovr(probs: np.ndarray, y_true: np.ndarray, class_names: list[str], out_png: Path, title: str = "PR Curves (OvR)"):
    if probs is None or probs.size == 0:
        return
    _ensure_parent(out_png)
    n_classes = probs.shape[1]
    y = np.eye(n_classes)[y_true]
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve(y[:, c], probs[:, c])
        ap = average_precision_score(y[:, c], probs[:, c])
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def per_class_specificity_sensitivity(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cm.shape[0]
    sens = np.zeros(n, dtype=float)
    spec = np.zeros(n, dtype=float)
    total = cm.sum()
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - (tp + fn + fp)
        sens[i] = tp / (tp + fn) if (tp + fn) else 0.0
        spec[i] = tn / (tn + fp) if (tn + fp) else 0.0
    return spec, sens

# ---------- Calibration helpers ----------
def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    if probs is None or probs.size == 0 or len(y_true) == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None, class_names: list[str] | None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1) if not class_names else len(class_names)
    names = class_names if class_names else [f"class_{i}" for i in range(n_classes)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    spec, sens = per_class_specificity_sensitivity(cm)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    auc_ovr = None
    ap_ovr = None
    if probs is not None and probs.size and probs.shape[1] == n_classes:
        try:
            auc_ovr = float(roc_auc_score(y_true, probs, multi_class="ovr"))
        except Exception:
            auc_ovr = None
        try:
            ap_ovr = float(average_precision_score(pd.get_dummies(y_true).reindex(columns=range(n_classes), fill_value=0).values, probs, average="macro"))
        except Exception:
            ap_ovr = None
    ece = expected_calibration_error(probs, y_true, n_bins=10) if probs is not None else float("nan")
    per_class = []
    for i in range(n_classes):
        per_class.append({"class_id": i, "class_name": names[i], "precision": float(prec[i]), "recall_sensitivity": float(rec[i]), "f1": float(f1[i]), "specificity": float(spec[i]), "support": int(sup[i])})
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_precision": float(prec_macro), "macro_recall_sensitivity": float(rec_macro), "macro_f1": float(f1_macro), "micro_precision": float(prec_micro), "micro_recall": float(rec_micro), "micro_f1": float(f1_micro), "auc_roc_ovr": auc_ovr, "avg_precision_ovr": ap_ovr, "ece": ece, "confusion_matrix": cm.tolist(), "per_class": per_class}
    return summary

def report_tables(report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class_df = pd.DataFrame(report["per_class"])
    summary_df = pd.DataFrame([{k: report[k] for k in report.keys() if k not in {"per_class", "confusion_matrix"}}])
    return summary_df, per_class_df


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix_counts_and_percent(cm: np.ndarray, class_names: list[str], out_png: Path, title: str = "Confusion matrix", cmap: str = "viridis"):
    cm = np.asarray(cm)
    _ensure_dir(out_png)
    row_sum = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_pct = (cm / row_sum) * 100.0
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i,j])}\n({cm_pct[i,j]:.1f}%)"
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(cm_pct, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=100, xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "% of true class"})
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_confusion_and_classification_report(y_true, y_pred, class_names, out_png, title="Classification Report"):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6,1])
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)", ha="center", va="center", color="black", fontsize=10)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted label", ylabel="True label",
           title="Confusion Matrix")
    ax.set_aspect('equal')                     
    ax.set_xlim(-0.5, len(class_names)-0.5)   
    ax.set_ylim(len(class_names)-0.5, -0.5)  
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="% of true class")
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    
    df = pd.DataFrame(classification_report( y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)).T
    spec = {}
    for i, c in enumerate(class_names):
        TP = cm[i,i]; FP = cm[:,i].sum() - TP; FN = cm[i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec[c] = TN / (TN + FP) if (TN + FP) else 0
    df["specificity"] = np.nan
    for c, v in spec.items(): df.loc[c, "specificity"] = v
    cols = ["precision", "recall", "f1-score", "specificity", "support"]
    df = df.reindex(columns=cols)
    df["support"] = df["support"].round(0).astype("Int64")
    df = df.round(2)
    df = df.fillna("")
    df = df.astype(object) 
    if "accuracy" in df.index:
        df.loc["accuracy", ["precision", "recall", "specificity", "support"]] = ""
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); 
    tbl.scale(1.2,1.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white"); cell.set_facecolor("#123456")
        else:
            cell.set_facecolor("#E5EFF8" if r % 2 == 0 else "white")
            #cell.set_facecolor("#F8FBFD" )
        cell.set_edgecolor("#F8FBFD"); 
        cell.set_linewidth(0.2)
    for c in range(df.shape[1]):
        for r in range(1, df.shape[0] + 1): tbl[(r, c)]._loc = "right"
    ax.set_title("Classification Report", fontsize=10, pad=0)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); 
    plt.savefig(out_png, dpi=300); 
    plt.close()

def plot_misclassifications_bar(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_png: Path, title: str = "Misclassifications by true class"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _ensure_dir(out_png)
    mis_mask = (y_true != y_pred)
    counts = np.bincount(y_true[mis_mask], minlength=len(class_names))
    df = pd.DataFrame({"class": class_names, "misclassified": counts})
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x="class", y="misclassified")
    ax.set_title(title)
    ax.set_xlabel("True class")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_reliability_diagram_multiclass(probs: np.ndarray, y_true: np.ndarray, out_png: Path, n_bins: int = 10, title: str = "Reliability diagram (max prob)"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    if probs.size == 0 or len(y_true) == 0:
        return
    _ensure_dir(out_png)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"confidence": conf[mask].mean(), "accuracy": acc[mask].mean(), "count": int(mask.sum())})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("confidence")
    plt.figure(figsize=(7, 6))
    ax = sns.lineplot(data=df, x="confidence", y="accuracy", marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_roc_ovr_multiclass(probs: np.ndarray, y_true: np.ndarray, class_names: list[str], out_png: Path, title: str = "ROC curves (OvR)"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    if probs.size == 0:
        return
    _ensure_dir(out_png)
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], probs[:, c])
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_pr_ovr_multiclass(probs: np.ndarray, y_true: np.ndarray, class_names: list[str], out_png: Path, title: str = "Precision-Recall curves (OvR)"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    if probs.size == 0:
        return
    _ensure_dir(out_png)
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_onehot[:, c], probs[:, c])
        ap = average_precision_score(y_onehot[:, c], probs[:, c])
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
