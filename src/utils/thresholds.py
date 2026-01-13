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
