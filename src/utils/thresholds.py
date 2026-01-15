from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ---------- Threshold sweeps ----------
def threshold_sweep_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        y_pred = (prob_pos >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({"threshold": float(t), "precision": float(precision), "recall_sensitivity": float(recall), "specificity": float(specificity), "f1": float(f1)})
    return pd.DataFrame(rows)

def threshold_sweep_ovr(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    n_classes = probs.shape[1]
    for c in range(n_classes):
        df = threshold_sweep_binary((y_true == c).astype(int), probs[:, c])
        df.insert(0, "class_id", c)
        df.insert(1, "class_name", class_names[c] if class_names else str(c))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def pick_operating_point(df: pd.DataFrame, min_sensitivity: float | None = None, min_specificity: float | None = None, objective: str = "max_f1") -> pd.Series:
    out = df.copy()
    if min_sensitivity is not None:
        out = out[out["recall_sensitivity"] >= float(min_sensitivity)]
    if min_specificity is not None:
        out = out[out["specificity"] >= float(min_specificity)]
    if len(out) == 0:
        return df.sort_values("f1", ascending=False).iloc[0]
    if objective == "max_f1":
        return out.sort_values("f1", ascending=False).iloc[0]
    if objective == "max_sensitivity":
        return out.sort_values(["recall_sensitivity", "specificity"], ascending=False).iloc[0]
    if objective == "max_specificity":
        return out.sort_values(["specificity", "recall_sensitivity"], ascending=False).iloc[0]
    return out.sort_values("f1", ascending=False).iloc[0]
import pandas as pd

def pick_operating_points_ovr( sweep_ovr: pd.DataFrame, min_sensitivity: dict,
    objective: str = "max_specificity", classes: list | None = None, min_precision: dict | None = None):
    df = sweep_ovr.copy()
    for col in ["threshold", "precision", "recall_sensitivity", "specificity", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if classes is not None:
        df = df[df["class_name"].isin(classes)].copy()
    ops = []
    for cname, g in df.groupby("class_name", sort=False):
        gg = g.dropna(subset=["threshold", "precision", "recall_sensitivity", "specificity", "f1"]).copy()
        # sensitivity constraint
        ms = min_sensitivity.get(cname)
        if ms is not None:
            gg = gg[gg["recall_sensitivity"] >= ms]
        if min_precision is not None and cname in min_precision:
            gg = gg[gg["precision"] >= min_precision[cname]]

        # if constraints impossible, fallback to best recall
        if len(gg) == 0:
            gg = g.sort_values(["recall_sensitivity", "threshold"], ascending=[False, False]).head(1)

        # objective + conservative tie-breaker: prefer higher threshold
        if objective == "max_specificity":
            row = gg.sort_values(["specificity", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
        elif objective == "max_f1":
            row = gg.sort_values(["f1", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        else:
            row = gg.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        ops.append(row)
    return pd.DataFrame(ops)

# Automated Policy Decision : 
'''
    we start from high risk first: TB; fall back class : Normal 
    Rule as follows: 
    if P(TB) >= TB_threshold then classicify as TB
    elif P(Pneumonia) >= Pneumonia_threshold -> Pneumonia
    else: Normal
'''
def predict_with_policy( probs_all: np.ndarray, names: list, thresholds: dict,
    priority=("Tuberculosis", "Pneumonia"), fallback="Normal"):
    name_to_idx = {n: i for i, n in enumerate(names)}
    fallback_idx = name_to_idx[fallback]
    y_pred = np.full(probs_all.shape[0], fallback_idx, dtype=int)
    for cls in priority:
        cls_idx = name_to_idx[cls]
        thr = thresholds[cls]
        mask = (y_pred == fallback_idx) & (probs_all[:, cls_idx] >= thr)
        y_pred[mask] = cls_idx
    return y_pred

def predict_with_policy_explicit(probs_all, names, thr):
    idx = {n:i for i,n in enumerate(names)}
    y_pred = np.full(len(probs_all), idx["Normal"], dtype=int)
    tb_mask = probs_all[:, idx["Tuberculosis"]] >= thr["Tuberculosis"]
    y_pred[tb_mask] = idx["Tuberculosis"]
    pna_mask = (
        (y_pred == idx["Normal"]) &
        (probs_all[:, idx["Pneumonia"]] >= thr["Pneumonia"]))
    y_pred[pna_mask] = idx["Pneumonia"]
    return y_pred


