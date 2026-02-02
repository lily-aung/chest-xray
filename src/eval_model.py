from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import mlflow
import argparse
from src.train_hog import build_split
import joblib

import matplotlib.pyplot as plt
from src.utils.config import load_config, get_config_for_backbone
from src.utils.mlflow_utils import get_run_by_id , load_run_model_pytorch
from src.data.dataset import build_test_dataset
from src.data.dataloaders import build_dataloader
from src.utils.eval_plots import *
from src.utils.thresholds import *
from src.utils.best_model_selector import select_best_from_summaries

@torch.no_grad()
def predict_torch(model, loader, device):
    model.eval().to(device)
    y_true, y_pred, probs_max, probs_all, paths = [], [], [], [], []
    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
            batch_paths = [""] * len(labels)
        else:
            images, labels, batch_paths = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(p, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        probs_max.extend(p.max(dim=1).values.detach().cpu().numpy().tolist())
        probs_all.append(p.detach().cpu().numpy())
        paths.extend(list(batch_paths))
    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 0))
    return np.array(y_true), np.array(y_pred), np.array(probs_max), probs_all, paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--out_root", type=str, default="reports/best_model_eval", help="Output directory")
    args = parser.parse_args() 
    #out_root = Path("reports/best_model_eval")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    #1) Select best models
    #best_cnn = get_best_run_from_experiment(experiment_name="cnn-ablation-analysis", metric="val_accuracy", maximize=True)
    best_cnn = get_run_by_id("6781800cf9dc4d2db96cade70720699f") #("aed54057967b410685676f118774eafa")
    print("\n===  CNN ===\n", best_cnn.to_string())
    cnn_run_id = best_cnn["run_id"]
    cnn_run_name = best_cnn["run_name"]
    # ---------- 2) Load configs (per-model) ----------
    cnn_cfg_path = "configs/cnn_baseline_v2.yaml"

    cnn_cfg = load_config(cnn_cfg_path)

    cnn_class_names = getattr(cnn_cfg.data, "class_names", None) or getattr(cnn_cfg, "class_names", None) or []

    # ---------- 3) Build test loaders (each model uses its own dataset/transforms) ----------
    test_csv = args.test  
    cnn_cfg.data.test_csv = test_csv

    cnn_test_ds = build_test_dataset(cnn_cfg, test_csv)
    cnn_test_loader = build_dataloader(cnn_test_ds, batch_size=cnn_cfg.training.batch_size,
                                        shuffle=False, num_workers=cnn_cfg.data.num_workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("CNN test dataset len:", len(cnn_test_ds))

    # ---------- 4) Load models from MLflow and predict ----------

    # =========================
    # CNN Model: Load + Predict + Evaluate
    # =========================
    print(cnn_run_id)
    #cnn_model = load_run_model_pytorch(cnn_run_id, artifact_path="data")
    cnn_model = load_run_model_pytorch(cnn_run_id)
    print(cnn_test_loader)
    y_true, y_pred, y_conf, probs_all, paths = predict_torch(cnn_model, cnn_test_loader, device)
    cnn_out = out_root / f"cnn__{cnn_run_name}"
    cnn_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred, "confidence": y_conf}).to_csv(cnn_out / "predictions.csv", index=False)
    np.save(cnn_out / "probs_all.npy", probs_all)

    cnn_report = compute_classification_report(y_true, y_pred, probs_all, cnn_class_names)
    summary_df, per_class_df = report_tables(cnn_report)
    summary_df.to_csv(cnn_out / "summary_metrics.csv", index=False)
    per_class_df.to_csv(cnn_out / "per_class_metrics.csv", index=False)
    (cnn_out / "report.json").write_text(json.dumps(cnn_report, indent=2))

    names = cnn_class_names or [str(i) for i in range(probs_all.shape[1])]
    plot_confusion_and_classification_report(
        y_true, y_pred, names, cnn_out / "ConfusionMatrix_Classification.png", title="CNN Classification Report" )
    plot_misclassifications_bar(y_true, y_pred, names, cnn_out / "misclass_by_true_class.png")
    plot_roc_ovr_multiclass(probs_all, y_true, names, cnn_out / "roc_ovr.png")##x:FPR, y:TPR (Recall/sensiviity) ; AUC: higher better sep
    plot_pr_ovr_multiclass(probs_all, y_true, names, cnn_out / "pr_ovr.png")# X:Recall, Y:Precision
    plot_reliability_diagram_multiclass(probs_all, y_true, cnn_out / "reliability.png")
    thr = None
    if probs_all.shape[1] == 2:
        sweep = threshold_sweep_binary(y_true, probs_all[:, 1])
        sweep.to_csv(cnn_out / "threshold_sweep.csv", index=False)
        op = pick_operating_point(sweep, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op.to_dict()]).to_csv(cnn_out / "operating_point.csv", index=False)
    else:
        sweep = threshold_sweep_ovr(y_true, probs_all, names)  #Onevsrest
        sweep.to_csv(cnn_out / "threshold_sweep_ovr.csv", index=False)
        ops = pick_operating_points_ovr( sweep, classes=["Tuberculosis", "Pneumonia"],objective="max_specificity",
            min_sensitivity={"Tuberculosis": 0.90, "Pneumonia": 0.85}, min_precision={"Tuberculosis": 0.70})
        print(ops.to_string(index=False))
        
        ops.to_csv(cnn_out / "operating_points_ovr.csv", index=False) 
        thr = {r["class_name"]: float(r["threshold"]) for _, r in ops.iterrows()}
        print("Chosen thresholds:", thr)
        (cnn_out / "thresholds_policy.json").write_text(json.dumps(thr, indent=2))

        # apply policy
        y_pred_policy = predict_with_policy( probs_all, names, thr,priority=("Tuberculosis", "Pneumonia"), fallback="Normal")

        plot_confusion_and_classification_report(  y_true, y_pred_policy, names, cnn_out / "ConfusionMatrix_Classification_Policy.png",
            title="CNN Classification Report (Policy)")

        policy_report = compute_classification_report(y_true, y_pred_policy, probs_all, names)
        summary_pol, per_class_pol = report_tables(policy_report)
        summary_pol.to_csv(cnn_out / "summary_metrics_policy.csv", index=False)
        per_class_pol.to_csv(cnn_out / "per_class_metrics_policy.csv", index=False)
        (cnn_out / "report_policy.json").write_text(json.dumps(policy_report, indent=2))

    best = select_best_from_summaries(out_root, metric="macro_f1", prefer="policy")
    meta = { "best_model": best}
    if best["model_name"].startswith("cnn__"):
        best["mlflow_run_id"] = cnn_run_id
        best["model_family"] = "cnn"
        #best["gradcam_target_layer"] = "features.8"
        best["gradcam_target_layer"] = "features.12"
    
    meta_path = out_root / "best_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n[Best Model Selected]")
    print(f"[Saved] {meta_path}")
if __name__ == "__main__":
    main()
