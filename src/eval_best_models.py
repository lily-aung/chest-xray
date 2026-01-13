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

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, average_precision_score
import matplotlib.pyplot as plt
from src.utils.config import load_config, get_config_for_backbone
from src.utils.mlflow_utils import get_best_run_from_experiment, get_best_linear_probe_runs, load_hog_summary, load_run_model_pytorch
from src.data.dataset import build_test_dataset
from src.data.dataloaders import build_dataloader
from src.utils.eval_plots import *
from src.utils.thresholds import *
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
    args = parser.parse_args()  # <--- you MUST call parse_args()!

    out_root = Path("reports/best_model_eval")
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
    test_csv = args.test  # <-- now this works
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
    print(cnn_run_id)
    #cnn_model = load_run_model_pytorch(cnn_run_id, artifact_path="data")
    cnn_model = load_run_model_pytorch(cnn_run_id)
    print(cnn_test_loader)
    y_true, y_pred, y_conf, probs_all, paths = predict_torch(cnn_model, cnn_test_loader, device)
    cnn_out = out_root / f"cnn__{cnn_run_name}"
    cnn_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred, "confidence": y_conf}).to_csv(cnn_out / "predictions.csv", index=False)
    
    cnn_report = compute_classification_report(y_true, y_pred, probs_all, cnn_class_names)
    summary_df, per_class_df = report_tables(cnn_report)
    summary_df.to_csv(cnn_out / "summary_metrics.csv", index=False)
    per_class_df.to_csv(cnn_out / "per_class_metrics.csv", index=False)
    (cnn_out / "report.json").write_text(json.dumps(cnn_report, indent=2))

    names = cnn_class_names or [str(i) for i in range(probs_all.shape[1])]
    cm = np.array(cnn_report["confusion_matrix"])
    
    plot_confusion_matrix_counts_and_percent(cm, names, cnn_out / "cm_counts_percent.png", title="CNN Probe Confusion matrix")
    plot_misclassifications_bar(y_true, y_pred, names, cnn_out / "misclass_by_true_class.png")
    plot_reliability_diagram_multiclass(probs_all, y_true, cnn_out / "reliability.png")
    plot_roc_ovr_multiclass(probs_all, y_true, names, cnn_out / "roc_ovr.png")
    plot_pr_ovr_multiclass(probs_all, y_true, names, cnn_out / "pr_ovr.png")


    if probs_all.shape[1] == 2:
        sweep = threshold_sweep_binary(y_true, probs_all[:, 1])
        sweep.to_csv(cnn_out / "threshold_sweep.csv", index=False)
        op = pick_operating_point(sweep, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op.to_dict()]).to_csv(cnn_out / "operating_point.csv", index=False)
    else:
        sweep = threshold_sweep_ovr(y_true, probs_all, names)
        sweep.to_csv(cnn_out / "threshold_sweep_ovr.csv", index=False)

    # ---------- Linear Probe winner ----------
    #lp_model = load_run_model_pytorch(lp_run_id, artifact_path="data")
    lp_model = load_run_model_pytorch(lp_run_id)
    y_true, y_pred, y_conf, probs_all, paths = predict_torch(lp_model, lp_test_loader, device)
    lp_out = out_root / f"linear_probe__{lp_run_name}"
    lp_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred, "confidence": y_conf}).to_csv(lp_out / "predictions.csv", index=False)
    
    lp_report = compute_classification_report(y_true, y_pred, probs_all, lp_class_names)
    summary_df, per_class_df = report_tables(lp_report)
    summary_df.to_csv(lp_out / "summary_metrics.csv", index=False)
    per_class_df.to_csv(lp_out / "per_class_metrics.csv", index=False)
    (lp_out / "report.json").write_text(json.dumps(lp_report, indent=2))

    names = lp_class_names or [str(i) for i in range(probs_all.shape[1])]
    cm = np.array(lp_report["confusion_matrix"])
    plot_confusion_matrix_counts_and_percent(cm, names, lp_out / "cm_counts_percent.png", title="Linear Probe Confusion matrix")
    plot_misclassifications_bar(y_true, y_pred, names, lp_out / "misclass_by_true_class.png")
    plot_reliability_diagram_multiclass(probs_all, y_true, lp_out / "reliability.png")
    plot_roc_ovr_multiclass(probs_all, y_true, names, lp_out / "roc_ovr.png")
    plot_pr_ovr_multiclass(probs_all, y_true, names, lp_out / "pr_ovr.png")


    if probs_all.shape[1] == 2:
        sweep = threshold_sweep_binary(y_true, probs_all[:, 1])
        sweep.to_csv(lp_out / "threshold_sweep.csv", index=False)
        op = pick_operating_point(sweep, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op.to_dict()]).to_csv(lp_out / "operating_point.csv", index=False)
    else:
        sweep = threshold_sweep_ovr(y_true, probs_all, names)
        sweep.to_csv(lp_out / "threshold_sweep_ovr.csv", index=False)

    # ----------------- HOG winner: full inference at eval time -----------------
    hog_best = hog.sort_values("val_macro_f1", ascending=False).iloc[0].to_dict()

    hog_dir = Path("artifacts/hog_baselines")
    hog_model_name = hog_best.get("model", "xgb")  # "xgb" / "rf" / "mlp"
    hog_model_path = hog_dir / f"{hog_model_name}_model.joblib"
    if not hog_model_path.exists():
        raise FileNotFoundError(f"HOG model not found: {hog_model_path}")

    hog_model = joblib.load(hog_model_path)

    # HOG params from summary.csv
    img_size = int(hog_best.get("img_size", 224)) if "img_size" in hog_best else 224
    clahe_clip = float(hog_best.get("clahe_clip", 2.0))
    grid_str = str(hog_best.get("clahe_grid", "(8, 8)")).replace("(", "").replace(")", "")
    grid = tuple(int(x.strip()) for x in grid_str.split(","))
    hog_ppc = int(hog_best.get("hog_ppc", 16))
    hog_cpb = int(hog_best.get("hog_cpb", 2))
    hog_orientations = int(hog_best.get("hog_orientations", 9))

    # Build HOG features : testset
    Xte, yte, pte = build_split(csv_path=test_csv, img_root=None, img_col='image_path',
                                 label_col='label', img_size=img_size, clip=clahe_clip, grid=grid, orientations=hog_orientations,
                                   ppc=hog_ppc, cpb=hog_cpb, augment_train=False)
    y_pred = hog_model.predict(Xte)
    proba = hog_model.predict_proba(Xte) if hasattr(hog_model, "predict_proba") else None
    hog_out = out_root / f"hog__{hog_model_name}"
    hog_out.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({"path": pte, "y_true": yte, "y_pred": y_pred})
    if proba is not None:
        for k in range(proba.shape[1]):
            df_pred[f"proba_class_{k}"] = proba[:, k]
        df_pred["confidence"] = proba.max(axis=1)
    df_pred.to_csv(hog_out / "predictions.csv", index=False)
    proba_all = (np.eye(int(max(yte.max(), y_pred.max())) + 1)[y_pred]).astype(float) if proba is None else proba
    hog_class_names = cnn_class_names if cnn_class_names else [f"class_{i}" for i in range(proba_all.shape[1])]
    hog_report = compute_classification_report(np.array(yte), np.array(y_pred), np.array(proba_all), hog_class_names)
    summary_df, per_class_df = report_tables(hog_report)
    summary_df.to_csv(hog_out / "summary_metrics.csv", index=False)
    per_class_df.to_csv(hog_out / "per_class_metrics.csv", index=False)
    (hog_out / "report.json").write_text(json.dumps(hog_report, indent=2))

    names = hog_class_names or [str(i) for i in range(np.array(proba_all).shape[1])]
    cm = np.array(hog_report["confusion_matrix"])

    plot_confusion_matrix_counts_and_percent(cm, names, hog_out / "cm_counts_percent.png", title="HOG Confusion matrix")
    plot_misclassifications_bar(y_true, y_pred, names, hog_out / "misclass_by_true_class.png")
    plot_reliability_diagram_multiclass(probs_all, y_true, hog_out / "reliability.png")
    plot_roc_ovr_multiclass(probs_all, y_true, names, hog_out / "roc_ovr.png")
    plot_pr_ovr_multiclass(probs_all, y_true, names, hog_out / "pr_ovr.png")

    if np.array(proba_all).shape[1] == 2:
        sweep = threshold_sweep_binary(np.array(yte), np.array(proba_all)[:, 1])
        sweep.to_csv(hog_out / "threshold_sweep.csv", index=False)
        op = pick_operating_point(sweep, min_sensitivity=0.90, objective="max_specificity")
        pd.DataFrame([op.to_dict()]).to_csv(hog_out / "operating_point.csv", index=False)
    else:
        sweep = threshold_sweep_ovr(np.array(yte), np.array(proba_all), names)
        sweep.to_csv(hog_out / "threshold_sweep_ovr.csv", index=False)


if __name__ == "__main__":
    main()
