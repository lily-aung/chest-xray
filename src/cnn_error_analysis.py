from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_class_names(names_csv: str) -> list[str]:
    names = [x.strip() for x in names_csv.split(",") if x.strip()]
    if not names:
        raise ValueError("--names is empty. Provide something like: Normal,Pneumonia,Tuberculosis")
    return names
def parse_policy_thresholds(thr_arg: str | None) -> dict[str, float] | None:
    if not thr_arg:
        return None
    p = Path(thr_arg)
    if p.exists():
        obj = json.loads(p.read_text())
    else:
        obj = json.loads(thr_arg)
    if not isinstance(obj, dict):
        raise ValueError("policy_thresholds must be a JSON object like {'Tuberculosis':0.18, 'Pneumonia':0.51}")

    out: dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception as e:
            raise ValueError(f"Invalid threshold value for '{k}': {v}") from e
    return out

def ensure_required_cols(df: pd.DataFrame, required: set[str], where: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{where} missing required columns={sorted(missing)}. Got columns={list(df.columns)}")

def write_case_list(out_dir: Path, filename: str, paths: list[Any], max_items: int | None = None) -> Path:

    clean = [str(p) for p in paths if p is not None and str(p).strip() != ""]
    # unique while preserving order
    clean = list(dict.fromkeys(clean))
    if max_items is not None:
        clean = clean[: int(max_items)]

    out_path = out_dir / filename
    out_path.write_text(json.dumps({"paths": clean}, indent=2))
    return out_path


# =============================================================================
# Analysis helpers
def confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    Table of counts and rates for each true->pred pair.
    """
    rows: list[dict[str, Any]] = []
    n = int(len(y_true))

    for true_id, true_name in enumerate(names):
        mask_t = (y_true == true_id)
        support_true = int(mask_t.sum())

        for pred_id, pred_name in enumerate(names):
            cnt = int(((y_pred == pred_id) & mask_t).sum())
            rate_within_true = (cnt / support_true) if support_true > 0 else 0.0
            rows.append({"true_id": true_id, "true_name": true_name, "pred_id": pred_id, "pred_name": pred_name, 
                         "count": cnt, "rate_within_true": float(rate_within_true), "support_true": support_true})

    df = pd.DataFrame(rows).sort_values(["true_id", "pred_id"]).reset_index(drop=True)
    df["overall_rate"] = df["count"] / max(n, 1)
    return df


def extract_cases(df: pd.DataFrame, names: list[str], true_name: str, pred_name: str, *, top_k: int = 50, 
                  sort_by: str = "confidence", ascending: bool = True) -> pd.DataFrame:

    """
    Filter df to (y_true=true_name and y_pred=pred_name).
    sort_by:
      - "confidence" typical: lower confidence first -> ascending=True
      - any numeric column

    Note: "confidence" should mean p(predicted_class) or similar.
    """
    if true_name not in names or pred_name not in names:
        return pd.DataFrame()

    t = names.index(true_name)
    p = names.index(pred_name)
    sub = df[(df["y_true"] == t) & (df["y_pred"] == p)].copy()
    if sub.empty:
        return sub

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=ascending)

    return sub.head(int(top_k))


def probability_summary(probs: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    Per-class prob distribution summary.
    """
    rows = []
    for i, cname in enumerate(names):
        p = probs[:, i]
        rows.append( {"class_name": cname, "p_mean": float(np.mean(p)),
                "p_median": float(np.median(p)),
                "p_p90": float(np.quantile(p, 0.90)), "p_p99": float(np.quantile(p, 0.99))})
    return pd.DataFrame(rows)

def find_borderline(df: pd.DataFrame, probs: np.ndarray, names: list[str], thresholds: dict[str, float], *, margin: float = 0.05, top_k: int = 200) -> pd.DataFrame:

    """
    Borderline = any class probability within [thr-margin, thr+margin] for policy classes.

    Output includes:
      - class_name, p_class, threshold, dist
      - plus path / y_true / y_pred for inspection
    """
    out: list[dict[str, Any]] = []

    for cname, thr in thresholds.items():
        if cname not in names:
            continue

        cid = names.index(cname)
        p = probs[:, cid]

        idxs = np.where((p >= thr - margin) & (p <= thr + margin))[0]
        for k in idxs:
            row = df.iloc[int(k)]
            out.append({"row": int(k), "path": str(row["path"]), "y_true": int(row["y_true"]), 
                        "y_pred": int(row["y_pred"]), "class_name": cname, "p_class": float(p[k]), 
                        "threshold": float(thr), "dist": float(abs(p[k] - thr)), 
                        "confidence": float(row["confidence"]) if "confidence" in df.columns else np.nan})

    out_df = pd.DataFrame(out)
    if out_df.empty:
        return out_df

    # Closest-to-threshold first
    out_df = out_df.sort_values(["class_name", "dist"], ascending=[True, True]).head(int(top_k))
    return out_df.reset_index(drop=True)


# =============================================================================
# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(  "--run_dir", type=str, required=True,
        help="Path to reports/.../cnn__RUNNAME directory that contains predictions.csv and probs_all.npy")
    
    parser.add_argument( "--names", type=str,  default="Normal,Pneumonia,Tuberculosis",
        help="Comma-separated class names in correct index order.", )
    
    parser.add_argument( "--probs_npy", type=str, default=None,
        help="Optional explicit path to probs_all.npy. If omitted, uses run_dir/probs_all.npy" )
    
    parser.add_argument(  "--policy_thresholds", type=str, default=None,
        help='Optional JSON string or JSON file path. Example: \'{"Tuberculosis":0.18,"Pneumonia":0.51}\'')
    parser.add_argument("--borderline_margin", type=float, default=0.05)

    parser.add_argument("--viz_topk", type=int, default=100, help="How many paths per viz bucket JSON.")
    parser.add_argument("--high_risk_topk", type=int, default=100, help="Rows per high-risk error in CSV export.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = parse_class_names(args.names)
    name_to_id = {n: i for i, n in enumerate(names)}

    # -------------------------
    # Load predictions
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    df = pd.read_csv(pred_path)
    ensure_required_cols(df, {"path", "y_true", "y_pred"}, "predictions.csv")

    # confidence is ideally p(y_pred) or max probability
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)

    # -------------------------
    # Load probabilities
    probs_path = Path(args.probs_npy) if args.probs_npy else (run_dir / "probs_all.npy")
    if not probs_path.exists():
        raise FileNotFoundError( f"Missing probs array at {probs_path}. "
            f"Save it during eval: np.save(run_dir/'probs_all.npy', probs_all)")

    probs = np.load(probs_path)
    if probs.ndim != 2 or probs.shape[1] != len(names):
        raise ValueError(
            f"probs shape mismatch. Expected (N,{len(names)}) for names={names}, got {tuple(probs.shape)}")

    # -------------------------
    # 1) Confusion taxonomy
    tax = confusion_pairs(y_true, y_pred, names)
    tax.to_csv(out_dir / "error_taxonomy.csv", index=False)

    # -------------------------
    # 2) High-risk errors (policy / screening)
    # Most critical is TB->Normal, then TB->Pneumonia, then Pneumonia->Normal.
    # Sort low-confidence first (more ambiguous / fragile decisions).
    high_risk_blocks = [
        ("Tuberculosis", "Normal"),
        ("Tuberculosis", "Pneumonia"),
        ("Pneumonia", "Normal")]

    high_risk_parts = []
    for tname, pname in high_risk_blocks:
        sub = extract_cases(df, names, tname, pname, top_k=int(args.high_risk_topk), 
                            sort_by="confidence" if "confidence" in df.columns else "path", ascending=True)
        if not sub.empty: high_risk_parts.append(sub.assign(error=f"{tname}→{pname}"))

    high_risk_df = pd.concat(high_risk_parts, axis=0, ignore_index=True) if high_risk_parts else pd.DataFrame()
    high_risk_df.to_csv(out_dir / "high_risk_cases.csv", index=False)

    # -------------------------
    # 3) Probability summary
    probability_summary(probs, names).to_csv(out_dir / "class_probability_summary.csv", index=False)

    # -------------------------
    # 4) Borderline near policy thresholds
    thresholds = parse_policy_thresholds(args.policy_thresholds)

    borderline = pd.DataFrame()
    if thresholds:
        borderline = find_borderline(df, probs, names, thresholds, margin=float(args.borderline_margin), top_k=200)
        borderline.to_csv(out_dir / "borderline_cases.csv", index=False)


    # -------------------------
    # 6) Viz buckets (JSON lists)
    #    Later will be used by Grad-CAM for analyis
    viz_dir = out_dir / "viz_cases"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 6.1) High-risk splits per error label
    if not high_risk_df.empty:
        for err_name, g in high_risk_df.groupby("error"):
            # e.g. "Tuberculosis→Normal" -> "tuberculosis_to_normal"
            fname = err_name.replace("→", "_to_").replace(" ", "").lower()
            write_case_list(viz_dir, f"high_risk__{fname}.json", g["path"].tolist(), max_items=args.viz_topk)

        # TB-specific convenience buckets
        tb_to_normal = high_risk_df[high_risk_df["error"] == "Tuberculosis→Normal"]
        if not tb_to_normal.empty:
            write_case_list(
                viz_dir,
                "tb_fn__tb_to_normal.json",
                tb_to_normal["path"].tolist(),
                max_items=args.viz_topk,
            )

        tb_to_pna = high_risk_df[high_risk_df["error"] == "Tuberculosis→Pneumonia"]
        if not tb_to_pna.empty:
            write_case_list(
                viz_dir,
                "tb_fn__tb_to_pneumonia.json",
                tb_to_pna["path"].tolist(),
                max_items=args.viz_topk,
            )

    # 6.2) Borderline splits per policy class (closest to threshold first)
    if thresholds and not borderline.empty:
        for cname in borderline["class_name"].unique():
            sub = borderline[borderline["class_name"] == cname].copy()
            sub = sub.sort_values("dist", ascending=True)
            fname = cname.replace(" ", "").lower()
            write_case_list(viz_dir, f"borderline__{fname}.json", sub["path"].tolist(), max_items=args.viz_topk)

        # TB-specific borderline bucket (if TB threshold exists)
        if "Tuberculosis" in thresholds:
            sub_tb = borderline[borderline["class_name"] == "Tuberculosis"].copy().sort_values("dist", ascending=True)
            if not sub_tb.empty:
                write_case_list(viz_dir, "tb_borderline.json", sub_tb["path"].tolist(), max_items=args.viz_topk)

    # 6.3) Correct + high-confidence exemplars per class (controls)
    correct_df = df[df["y_true"] == df["y_pred"]].copy()
    for cname in names:
        cid = name_to_id[cname]
        sub = correct_df[correct_df["y_true"] == cid].copy()
        if sub.empty:
            continue

        idx = sub.index.to_numpy()
        sub["p_true"] = probs[idx, cid]
        sub = sub.sort_values("p_true", ascending=False)

        fname = cname.replace(" ", "").lower()
        write_case_list(viz_dir, f"correct_highconf__{fname}.json", sub["path"].tolist(), max_items=args.viz_topk)

    # 6.4) False-positive TB buckets (Normal/Pneumonia -> TB)
    if "Tuberculosis" in name_to_id:
        tb_id = name_to_id["Tuberculosis"]
        fp_tb = df[df["y_pred"] == tb_id].copy()
        fp_tb = fp_tb[fp_tb["y_true"] != tb_id]

        if not fp_tb.empty:
            # rank by TB probability descending (most confident TB false alarmsP
            idx = fp_tb.index.to_numpy()
            fp_tb["p_tb"] = probs[idx, tb_id]
            fp_tb = fp_tb.sort_values("p_tb", ascending=False)

            # split by true class for inspection
            for true_name in ["Normal", "Pneumonia"]:
                if true_name not in name_to_id:
                    continue
                tid = name_to_id[true_name]
                sub = fp_tb[fp_tb["y_true"] == tid].copy()
                if sub.empty:
                    continue
                fname = true_name.replace(" ", "").lower()
                write_case_list(viz_dir, f"tb_fp__{fname}_to_tb.json", sub["path"].tolist(), max_items=args.viz_topk)

    # -------------------------
    # 5) Sample list for quick Grad-CAM 
    sample_paths: list[str] = []
    if not high_risk_df.empty:
        sample_paths += high_risk_df["path"].dropna().astype(str).head(30).tolist()
    if thresholds and not borderline.empty:
        sample_paths += borderline["path"].dropna().astype(str).head(30).tolist()
    sample_paths = list(dict.fromkeys(sample_paths))
    (out_dir / "sample_cases.json").write_text(json.dumps({"paths": sample_paths}, indent=2))
    print(f"[Final] Save analysis to: {out_dir}")

if __name__ == "__main__":
    main()
