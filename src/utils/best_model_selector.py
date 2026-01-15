from __future__ import annotations
from pathlib import Path
import pandas as pd
from pathlib import Path
import pandas as pd

def _read_metric(summary_csv: Path, metric: str) -> float | None:
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    if metric in df.columns and len(df) >= 1:
        try:
            return float(df.iloc[0][metric])
        except Exception:
            pass
    if {"metric", "value"}.issubset(df.columns):
        sub = df[df["metric"] == metric]
        if len(sub) > 0:
            try:
                return float(sub.iloc[0]["value"])
            except Exception:
                return None
    return None

def select_best_from_summaries( out_root: str | Path,
    metric: str = "macro_f1", prefer: str = "policy",
    allow_fallback_to_argmax: bool = True ) -> dict:
    """
    Args:
      out_root: reports/best_model_eval
      metric: "macro_f1", "accuracy", "micro_f1"
      prefer: "policy" or "argmax" 
      allow_fallback_to_argmax: if policy summary missing, use argmax summary
    Returns:
      dict with keys:
        model_name, run_dir, score, metric, source
    """
    out_root = Path(out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    candidates = []
    for run_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        model_name = run_dir.name

        policy_csv = run_dir / "summary_metrics_policy.csv"
        argmax_csv = run_dir / "summary_metrics.csv"

        policy_score = _read_metric(policy_csv, metric)
        argmax_score = _read_metric(argmax_csv, metric)

        # pick score based on preference
        source = None
        score = None

        if prefer == "policy":
            if policy_score is not None:
                score = policy_score
                source = "policy"
            elif allow_fallback_to_argmax and argmax_score is not None:
                score = argmax_score
                source = "argmax"
        else:  # prefer argmax
            if argmax_score is not None:
                score = argmax_score
                source = "argmax"
            elif allow_fallback_to_argmax and policy_score is not None:
                score = policy_score
                source = "policy"
        if score is None:
            continue
        candidates.append({ "model_name": model_name, "run_dir": str(run_dir),
            "score": float(score), "metric": metric, "source": source })

    if not candidates:
        raise RuntimeError(
            f"No candidates found under {out_root}. "
            f"Expected summary_metrics.csv or summary_metrics_policy.csv in each run folder.")
    # highest score wins
    best = sorted(candidates, key=lambda d: d["score"], reverse=True)[0]
    return best
