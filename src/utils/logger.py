import logging
import os
import mlflow
import wandb

from src.utils.mlflow_utils import log_metrics as log_mlflow_metrics
from src.utils.wandb_utils import log_metrics as log_wandb_metrics


def setup_logging(cfg):
    """
    Setup Python logger (file + optional console).
    Experiment lifecycle (MLflow/W&B init) must be handled outside.
    """
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    os.makedirs(cfg.logging.log_dir, exist_ok=True)

    log_file = os.path.join(cfg.logging.log_dir, "training.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console handler
    if cfg.logging.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def log_metrics(metrics: dict, step: int):
    """
    Log metrics to MLflow and WandB if active.

    Args:
        metrics (dict): {"train_loss": ..., "val_accuracy": ...}
        step (int): epoch or global step
    """
    if not metrics:
        return

    if mlflow.active_run():
        log_mlflow_metrics(metrics, step)

    if wandb.run:
        log_wandb_metrics(metrics, step)


def close_logging():
    """
    Close external logging backends safely.
    """
    if mlflow.active_run():
        mlflow.end_run()

    if wandb.run:
        wandb.finish()
