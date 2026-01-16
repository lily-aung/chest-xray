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
from src.utils.mlflow_utils import get_best_run_from_experiment, get_best_linear_probe_runs, load_hog_summary, load_run_model_pytorch
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
    best_cnn = get_best_run_from_experiment(experiment_name="cnn-ablation-analysis", metric="val_accuracy", maximize=True)
    best_lps = get_best_linear_probe_runs(experiment_name="backbone-linear-probe", metric="val_macro_f1", maximize=True)
    hog = load_hog_summary("artifacts/hog_baselines/summary.csv", metric="val_macro_f1")

    print("\n=== Best CNN ===\n", best_cnn.to_string())
    print("\n=== Best Linear Probes (per backbone) ===\n", best_lps.to_string(index=False))
    print("\n=== HOG Baselines ===\n", hog.to_string(index=False))

    # choose ONE LP winner overall (highest val_macro_f1)
    
    lp_winner = best_lps.sort_values("val_macro_f1", ascending=False).iloc[0]
    lp_backbone = lp_winner["backbone"]
    lp_run_id = lp_winner["run_id"]
    lp_run_name = lp_winner["run_name"]

    # CNN winner info
    cnn_run_id = best_cnn["run_id"]
    cnn_run_name = best_cnn["run_name"]

    # ---------- 2) Load configs (per-model) ----------
    cnn_cfg_path = "configs/cnn.yaml"
    lp_cfg_path = get_config_for_backbone(lp_backbone, config_dir="configs")

    cnn_cfg = load_config(cnn_cfg_path)
    lp_cfg = load_config(lp_cfg_path)

    cnn_class_names = getattr(cnn_cfg.data, "class_names", None) or getattr(cnn_cfg, "class_names", None) or []
    lp_class_names = getattr(lp_cfg.data, "class_names", None) or getattr(lp_cfg, "class_names", None) or []

    # ---------- 3) Build test loaders (each model uses its own dataset/transforms) ----------
    test_csv = args.test  
    cnn_cfg.data.test_csv = test_csv
    lp_cfg.data.test_csv = test_csv

    cnn_test_ds = build_test_dataset(cnn_cfg, test_csv)
    cnn_test_loader = build_dataloader(cnn_test_ds, batch_size=cnn_cfg.training.batch_size,
                                        shuffle=False, num_workers=cnn_cfg.data.num_workers)
    lp_test_ds = build_test_dataset(lp_cfg, test_csv)
    lp_test_loader = build_dataloader(lp_test_ds, batch_size=lp_cfg.training.batch_size, 
                                      shuffle=False, num_workers=lp_cfg.data.num_workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("CNN test dataset len:", len(cnn_test_ds))
    print("LP test dataset len:", len(lp_test_ds))

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

    # =========================
    # Linear Probe: Load + Predict + Evaluate
    # =========================
    print(lp_run_id)
    lp_model = load_run_model_pytorch(lp_run_id)
    print(lp_test_loader)
    y_true_lp, y_pred_lp, y_conf_lp, probs_all_lp, paths_lp = predict_torch(lp_model, lp_test_loader, device)
    lp_out = out_root / f"linearprobe__{lp_run_name}__{lp_backbone}"
    lp_out.mkdir(parents=True, exist_ok=True)
    lp_names = lp_class_names or [str(i) for i in range(probs_all_lp.shape[1])]
    assert len(lp_names) == probs_all_lp.shape[1], f"class_names mismatch: {lp_names} vs probs {probs_all_lp.shape}"

    pred_df_lp = pd.DataFrame({ "path": paths_lp, "y_true": y_true_lp, "y_pred": y_pred_lp, "confidence": y_conf_lp})
    probs_df_lp = pd.DataFrame(probs_all_lp, columns=[f"p_{n}" for n in lp_names])
    pd.concat([pred_df_lp, probs_df_lp], axis=1).to_csv(lp_out / "predictions.csv", index=False)
    np.save(lp_out / "probs_all.npy", probs_all_lp)
    lp_report = compute_classification_report(y_true_lp, y_pred_lp, probs_all_lp, lp_names)
    (lp_out / "report.json").write_text(json.dumps(lp_report, indent=2))

    plot_confusion_and_classification_report( y_true_lp, y_pred_lp, lp_names, lp_out / "ConfusionMatrix_Classification.png",
        title="Linear Probe Classification Report" )
    plot_misclassifications_bar(y_true_lp, y_pred_lp, lp_names, lp_out / "misclass_by_true_class.png")
    plot_roc_ovr_multiclass(probs_all_lp, y_true_lp, lp_names, lp_out / "roc_ovr.png")
    plot_pr_ovr_multiclass(probs_all_lp, y_true_lp, lp_names, lp_out / "pr_ovr.png")
    plot_reliability_diagram_multiclass(probs_all_lp, y_true_lp, lp_out / "reliability.png")

    thr_lp = None
    if probs_all_lp.shape[1] == 2:
        # binary case
        sweep_lp = threshold_sweep_binary(y_true_lp, probs_all_lp[:, 1])
        sweep_lp.to_csv(lp_out / "threshold_sweep.csv", index=False)
        op_lp = pick_operating_point(sweep_lp, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op_lp.to_dict()]).to_csv(lp_out / "operating_point.csv", index=False)

    else:   # multiclass OvR sweep
        sweep_lp = threshold_sweep_ovr(y_true_lp, probs_all_lp, lp_names)
        sweep_lp.to_csv(lp_out / "threshold_sweep_ovr.csv", index=False)

        # Select operating points for priority classes only (Normal is fallback)
        ops_lp = pick_operating_points_ovr( sweep_lp, classes=["Tuberculosis", "Pneumonia"],objective="max_specificity",
            min_sensitivity={"Tuberculosis": 0.90, "Pneumonia": 0.85},
            min_precision={"Tuberculosis": 0.70})
        print(ops_lp.to_string(index=False))
        ops_lp.to_csv(lp_out / "operating_points_ovr.csv", index=False)
        thr_lp = {r["class_name"]: float(r["threshold"]) for _, r in ops_lp.iterrows()}
        print("Chosen thresholds (LP):", thr_lp)
        (lp_out / "thresholds_policy.json").write_text(json.dumps(thr_lp, indent=2))

        # Apply policy (TB -> Pneumonia -> Normal)
        y_pred_policy_lp = predict_with_policy( probs_all_lp, lp_names, thr_lp, priority=("Tuberculosis", "Pneumonia"), fallback="Normal")

        # Plot policy CM/report
        plot_confusion_and_classification_report( y_true_lp, y_pred_policy_lp, lp_names,
            lp_out / "ConfusionMatrix_Classification_Policy.png",
            title="Linear Probe Classification Report (Policy)" )

        policy_report_lp = compute_classification_report(y_true_lp, y_pred_policy_lp, probs_all_lp, lp_names)
        summary_pol_lp, per_class_pol_lp = report_tables(policy_report_lp)
        summary_pol_lp.to_csv(lp_out / "summary_metrics_policy.csv", index=False)
        per_class_pol_lp.to_csv(lp_out / "per_class_metrics_policy.csv", index=False)
        (lp_out / "report_policy.json").write_text(json.dumps(policy_report_lp, indent=2))

    # ----------------- HOG winner: full inference at eval time -----------------
    # Build HOG features : testset

    hog_best = hog.sort_values("val_macro_f1", ascending=False).iloc[0].to_dict()
    hog_dir = Path("artifacts/hog_baselines")
    hog_model_name = hog_best.get("model", "xgb")  # "xgb" / "rf" / "mlp"
    hog_model_path = hog_dir / f"{hog_model_name}_model.joblib"
    if not hog_model_path.exists():
        raise FileNotFoundError(f"HOG model not found: {hog_model_path}")
    hog_model = joblib.load(hog_model_path)
    # HOG params extracted from summary.csv 
    img_size = int(hog_best.get("img_size", 224))
    clahe_clip = float(hog_best.get("clahe_clip", 2.0))

    grid_str = str(hog_best.get("clahe_grid", "(8, 8)")).replace("(", "").replace(")", "")
    grid = tuple(int(x.strip()) for x in grid_str.split(","))

    hog_ppc = int(hog_best.get("hog_ppc", 16))
    hog_cpb = int(hog_best.get("hog_cpb", 2))
    hog_orientations = int(hog_best.get("hog_orientations", 9))

    #  Build HOG features: test set 
    Xte, yte, pte = build_split(csv_path=test_csv, img_root=None, img_col='image_path',
                                    label_col='label', img_size=img_size, clip=clahe_clip, grid=grid, orientations=hog_orientations,
                                    ppc=hog_ppc, cpb=hog_cpb, augment_train=False)
    y_true_hog = np.array(yte)
    y_pred_hog = hog_model.predict(Xte)
    proba_hog = hog_model.predict_proba(Xte) if hasattr(hog_model, "predict_proba") else None
    if proba_hog is None:
        # fallback: one-hot of predictions (no real probabilities)
        n_classes = int(max(y_true_hog.max(), y_pred_hog.max())) + 1
        probs_all_hog = np.eye(n_classes)[y_pred_hog].astype(float)
        conf_hog = probs_all_hog.max(axis=1)
    else:
        probs_all_hog = np.array(proba_hog)
        conf_hog = probs_all_hog.max(axis=1)

    hog_out = out_root / f"hog__{hog_model_name}"
    hog_out.mkdir(parents=True, exist_ok=True)

    #  Class names 
    hog_class_names = cnn_class_names if cnn_class_names else [f"class_{i}" for i in range(probs_all_hog.shape[1])]
    names_hog = hog_class_names or [str(i) for i in range(probs_all_hog.shape[1])]
    assert len(names_hog) == probs_all_hog.shape[1], f"class_names mismatch: {names_hog} vs probs {probs_all_hog.shape}"

    #  Save predictions.csv (with per-class probabilities) 
    pred_df_hog = pd.DataFrame({ "path": pte, "y_true": y_true_hog,  "y_pred": y_pred_hog, "confidence": conf_hog })
    probs_df_hog = pd.DataFrame(probs_all_hog, columns=[f"p_{n}" for n in names_hog])
    pd.concat([pred_df_hog, probs_df_hog], axis=1).to_csv(hog_out / "predictions.csv", index=False)
    np.save(hog_out / "probs_all.npy", probs_all_hog)
    #  Base report ; threshodling using augmax
    hog_report = compute_classification_report(y_true_hog, y_pred_hog, probs_all_hog, names_hog)
    summary_df, per_class_df = report_tables(hog_report)
    summary_df.to_csv(hog_out / "summary_metrics.csv", index=False)
    per_class_df.to_csv(hog_out / "per_class_metrics.csv", index=False)
    (hog_out / "report.json").write_text(json.dumps(hog_report, indent=2))
    plot_confusion_and_classification_report( y_true_hog, y_pred_hog, names_hog,
        hog_out / "ConfusionMatrix_Classification.png",
        title=f"HOG ({hog_model_name}) Classification Report" )
    plot_misclassifications_bar(y_true_hog, y_pred_hog, names_hog, hog_out / "misclass_by_true_class.png")
    plot_roc_ovr_multiclass(probs_all_hog, y_true_hog, names_hog, hog_out / "roc_ovr.png")
    plot_pr_ovr_multiclass(probs_all_hog, y_true_hog, names_hog, hog_out / "pr_ovr.png")
    plot_reliability_diagram_multiclass(probs_all_hog, y_true_hog, hog_out / "reliability.png")

    thr_hog = None
    if probs_all_hog.shape[1] == 2:#binary
        sweep_hog = threshold_sweep_binary(y_true_hog, probs_all_hog[:, 1])
        sweep_hog.to_csv(hog_out / "threshold_sweep.csv", index=False)
        op_hog = pick_operating_point(sweep_hog, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op_hog.to_dict()]).to_csv(hog_out / "operating_point.csv", index=False)

    else: #multiclass 
        sweep_hog = threshold_sweep_ovr(y_true_hog, probs_all_hog, names_hog)
        sweep_hog.to_csv(hog_out / "threshold_sweep_ovr.csv", index=False)
        ops_hog = pick_operating_points_ovr(
            sweep_hog,
            classes=["Tuberculosis", "Pneumonia"],     # exclude Normal (fallback)
            objective="max_specificity",
            min_sensitivity={"Tuberculosis": 0.90, "Pneumonia": 0.85},
            min_precision={"Tuberculosis": 0.70})      # prevents TB threshold too low
        print(ops_hog.to_string(index=False))
        ops_hog.to_csv(hog_out / "operating_points_ovr.csv", index=False)
        thr_hog = {r["class_name"]: float(r["threshold"]) for _, r in ops_hog.iterrows()}
        print("Chosen thresholds (HOG):", thr_hog)
        (hog_out / "thresholds_policy.json").write_text(json.dumps(thr_hog, indent=2))

        y_pred_policy_hog = predict_with_policy( probs_all_hog, names_hog, thr_hog, 
                                                priority=("Tuberculosis", "Pneumonia"), fallback="Normal")

        plot_confusion_and_classification_report( y_true_hog, y_pred_policy_hog, names_hog,
            hog_out / "ConfusionMatrix_Classification_Policy.png",
            title=f"HOG ({hog_model_name}) Classification Report (Policy)")

        policy_report_hog = compute_classification_report(y_true_hog, y_pred_policy_hog, probs_all_hog, names_hog)
        summary_pol_hog, per_class_pol_hog = report_tables(policy_report_hog)
        summary_pol_hog.to_csv(hog_out / "summary_metrics_policy.csv", index=False)
        per_class_pol_hog.to_csv(hog_out / "per_class_metrics_policy.csv", index=False)
        (hog_out / "report_policy.json").write_text(json.dumps(policy_report_hog, indent=2))

    best = select_best_from_summaries(out_root, metric="macro_f1", prefer="policy")
    meta = { "best_model": best}
    if best["model_name"].startswith("cnn__"):
        best["mlflow_run_id"] = cnn_run_id
        best["model_family"] = "cnn"
        best["gradcam_target_layer"] = "features.8"
    elif best["model_name"].startswith("linearprobe__"):
        best["mlflow_run_id"] = lp_run_id
        best["model_family"] = "linear_probe"
    elif best["model_name"].startswith("hog__"):
        best["mlflow_run_id"] = None
        best["model_family"] = "hog"
    meta_path = out_root / "best_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n[Best Model Selected]")
    print(f"[Saved] {meta_path}")
if __name__ == "__main__":
    main()
