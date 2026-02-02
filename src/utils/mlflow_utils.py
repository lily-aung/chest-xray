from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def init_mlflow_bk(experiment_name, run_name, params=None):
    db_path = Path(__file__).resolve().parents[2] / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if params: mlflow.log_params(params)
    return run.info.run_id

def init_mlflow(mlflow_cfg, run_name, params=None):
    project_root = Path(__file__).resolve().parents[2]

    if not mlflow_cfg.enabled:
        return None

    # Tracking URI 
    tracking_uri = mlflow_cfg.tracking.uri
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        rel = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{(project_root / rel).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    artifact_root = (project_root / mlflow_cfg.artifacts.root).as_posix() # Artifact root 

    exp = mlflow.get_experiment_by_name(mlflow_cfg.experiment_name) # Experiment
    if exp is None:
        exp_id = mlflow.create_experiment(mlflow_cfg.experiment_name, artifact_location=artifact_root)
    else:
        exp_id = exp.experiment_id

    run = mlflow.start_run(run_name=run_name, experiment_id=exp_id)
    if params:  mlflow.log_params(params)

    # Tags 
    tags = _as_dict(getattr(mlflow_cfg, "tags", None) if not isinstance(mlflow_cfg, dict) else mlflow_cfg.get("tags"))
    for k, v in tags.items():
        mlflow.set_tag(str(k), str(v))
    return run.info.run_id

def log_metrics(metrics: dict, step: int):
    if not mlflow.active_run():
        return
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=int(step))

def log_model(model, artifact_path="model"):
    mlflow.pytorch.log_model(model, artifact_path=artifact_path)

def close_mlflow():
    if mlflow.active_run():
        mlflow.end_run()

def save_confusion_matrix_png(cm, class_names, out_path, title="Confusion Matrix"):
    import os, seaborn as sns, numpy as np, matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(cm, "detach"): cm = cm.detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.set_context("paper", font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=class_names, yticklabels=class_names,
                     square=True, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_confusion_matrix_png1(cm, class_names, out_path, title="Confusion Matrix"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = cm.detach().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_best_model_from_run(run_id: str, artifact_path: str, device):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.pytorch.load_model(model_uri)
    return model.to(device)

def _as_dict(x):
    if x is None: return {}
    if isinstance(x, dict): return x
    if hasattr(x, "__dict__"):
        return vars(x)
    return {}

# For test results comparison 
def get_best_run_from_experiment(experiment_name: str, metric: str = "val_macro_f1", maximize: bool = True) -> pd.Series:
    """Return the single best run from a CNN experiment."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None: 
        raise RuntimeError(f"Experiment not found: {experiment_name}")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
    mcol = f"metrics.{metric}"; runs = runs.dropna(subset=[mcol])
    best = runs.sort_values(mcol, ascending=not maximize).iloc[0]
    return pd.Series({"model": "cnn", "run_name": best.get("tags.mlflow.runName"), 
                      metric: best[mcol], "run_id": best["run_id"]})

def get_best_linear_probe_runs(experiment_name: str, metric: str = "macro_f1", maximize: bool = True) -> pd.DataFrame:
    """Return best linear probe per backbone """

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None: 
        raise RuntimeError(f"Experiment not found: {experiment_name}")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
    mcol = f"metrics.{metric}"; runs = runs.dropna(subset=[mcol])
    runs["backbone"] = runs["tags.mlflow.runName"].str.split("-lp").str[0]
    runs = runs[runs["backbone"] != "cnn-ft50ep-clahe"]###exclude from analyssi
    best = runs.sort_values(mcol, ascending=not maximize).groupby("backbone", as_index=False).head(1)
    return best[["backbone", "tags.mlflow.runName", mcol, "run_id"]].rename(
        columns={"tags.mlflow.runName": "run_name", mcol: metric})

def load_hog_summary(summary_csv: str | Path, metric: str = "macro_f1") -> pd.DataFrame:
    """Load HOG baseline results from CSV."""
    df = pd.read_csv(summary_csv)
    if metric not in df.columns: 
        raise RuntimeError(f"Metric '{metric}' not found in HOG summary. Available: {list(df.columns)}")
    #df["model"] = "hog"
    return df

# evaluate best
def get_best_run_id_by_metric( experiment_name: str, metric: str,
    maximize: bool = True ) -> Tuple[str, str, float]:
    """
    Return (run_id, run_name, metric_value) for the best run in MLflow experiment.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    mcol = f"metrics.{metric}"
    if mcol not in runs.columns:
        available = [c for c in runs.columns if c.startswith("metrics.")]
        raise RuntimeError(f"Metric '{metric}' not found in experiment '{experiment_name}'. Available: {available[:50]}")

    runs = runs.dropna(subset=[mcol]).sort_values(mcol, ascending=not maximize)
    best = runs.iloc[0]
    return best["run_id"], best.get("tags.mlflow.runName", ""), float(best[mcol])


def pick_first_existing_top_level_artifact(run_id: str, candidates: List[str]) -> str:
    """
    Return the first top-level artifact in candidates that exists in the run.
    """
    client = MlflowClient()
    top_level = {x.path for x in client.list_artifacts(run_id, path="")}
    for c in candidates:
        if c in top_level:
            return c
    raise RuntimeError(f"No candidate artifacts found in run {run_id}. Found: {sorted(top_level)}")


def download_artifact(run_id: str, artifact_path: str, dst_dir: Union[str, Path]) -> Path:
    """
    Download an artifact from a run to a local folder.
    """
    client = MlflowClient()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    local_path = client.download_artifacts(run_id, artifact_path, dst_dir.as_posix())
    return Path(local_path)

def _list_artifacts_recursive(run_id: str, path: str = "") -> list[str]:
    client = MlflowClient()
    out = []
    items = client.list_artifacts(run_id, path=path)
    for it in items:
        out.append(it.path)
        if it.is_dir:
            out.extend(_list_artifacts_recursive(run_id, it.path))
    return out


def find_mlflow_model_artifact_path(run_id: str) -> str:
    """Return MLflow model artifact path (contains MLmodel)"""
    client = MlflowClient()
    likely = ["best_model_macro_f1", "best_model", "model", "data"] # to handle different naming conventions
    for top in likely:
        try:
            items = client.list_artifacts(run_id, path=top)
        except Exception:
            continue
        for it in items:
            if it.path.endswith("MLmodel"):
                return top
        # <top>/data/MLmodel
        for it in items:
            if it.is_dir and it.path.endswith("data"):
                sub = client.list_artifacts(run_id, path=it.path)
                if any(x.path.endswith("MLmodel") for x in sub):
                    return it.path
    # fallback
    all_paths = _list_artifacts_recursive(run_id, "")
    mlmodel_files = [p for p in all_paths if p.endswith("MLmodel")]
    if not mlmodel_files:
        raise RuntimeError(
            f"No MLflow model found in run {run_id}. "
            f"Top-level artifacts: {[x.path for x in client.list_artifacts(run_id, '')]}")
    mlmodel_path = mlmodel_files[0]
    parent = "/".join(mlmodel_path.split("/")[:-1])
    return parent


def load_run_model_pytorch(run_id: str):
    """
      - <artifact>/MLmodel
      - <artifact>/data/MLmodel
    """
    model_path = find_mlflow_model_artifact_path(run_id)
    return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_path}")


def get_run_by_id(run_id: str, metric: str = "val_macro_f1") -> pd.Series:
    run = mlflow.get_run(run_id)

    metrics = run.data.metrics
    tags = run.data.tags

    if metric not in metrics:
        raise KeyError(f"Metric '{metric}' not found in run {run_id}")

    return pd.Series({
        "model": tags.get("model", "cnn"),
        "run_name": tags.get("mlflow.runName"),
        metric: metrics[metric],
        "run_id": run_id  })
