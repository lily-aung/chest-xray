import argparse
from pathlib import Path
import torch
import torch.optim.lr_scheduler as lr_scheduler
import mlflow
import mlflow.pytorch
import wandb

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logging, log_metrics, close_logging
from src.utils.mlflow_utils import init_mlflow
from src.utils.wandb_utils import init_wandb

from src.data.dataset import (build_datasets, build_test_dataset, infer_class_names_and_num_classes)
from src.data.dataloaders import build_dataloader

from src.models.build_model import build_model
from src.engine.trainer import Trainer
from src.engine.callbacks import GradientClippingCallback, LRSchedulerCallback
from src.utils.losses import (build_cross_entropy_with_weights, compute_class_weights_from_dataset)

def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--linear_probe", type=int, default=None)  # 1 or 0
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.training.seed = int(args.seed)
    if args.linear_probe is not None:
        cfg.model.linear_probe = bool(args.linear_probe)

    # Set seed once
    set_seed(cfg.training.seed)
    cfg_name = Path(args.config).stem
    mode = "probe" if getattr(cfg.model, "linear_probe", False) else "finetune"
    run_name = getattr(cfg.experiment, "run_name", None) or f"{cfg_name}-{mode}-seed{cfg.training.seed}"
    #cfg = load_config("configs/resnet50.yaml")
    #cfg = load_config("configs/efficientnet_b0.yaml")
    #cfg = load_config("configs/densenet121.yaml")
    #cfg = load_config("configs/swin_tiny_patch4_window7_224.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")
    mlflow_cfg = _cfg_get(mlflow_cfg, "mlflow", {}) 

    set_seed(cfg.training.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    print('devices on ', device)
    logger = setup_logging(cfg)
    run_name = cfg.experiment.run_name or f"seed-{cfg.training.seed}"
    # --- MLflow init
    mlflow_enabled = bool(_cfg_get(mlflow_cfg, "enabled", False)) or bool(getattr(cfg.logging, "use_mlflow", False))
    mlflow_run_id = None

    if mlflow_enabled:
        experiment_name = _cfg_get(mlflow_cfg, "experiment_name", None) or getattr(cfg.logging, "mlflow_experiment_name", "default")

        mlflow_run_id = init_mlflow( mlflow_cfg=mlflow_cfg, run_name=run_name,
            params={
                "batch_size": cfg.training.batch_size,
                "lr": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "seed": cfg.training.seed,
                "img_size": cfg.data.img_size,
                "weight_decay": cfg.training.weight_decay,
                "custom_vhe": cfg.augmentation.custom_clahe,
                "horizontal_flip": cfg.augmentation.horizontal_flip,
                "linear_probe": getattr(cfg.model, "linear_probe", False),
                "model_name": getattr(cfg.model, "name", "unknown"),},)
        mlflow.pytorch.autolog(log_models=False)
        logger.info(f"[MLflow] run_id={mlflow_run_id}")
    # --- W&B init ---
    if getattr(cfg.logging, "use_wandb", False):
        init_wandb(cfg.logging.wandb_project, run_name, config=cfg, mlflow_run_id=mlflow_run_id)

    # Sync IDs (MLflow ←→ W&B)
    if mlflow.active_run() and wandb.run:
        mlflow.set_tag("wandb_run_id", wandb.run.id)
        mlflow.set_tag("wandb_url", wandb.run.url)
    # --- Model ---
    model = build_model(cfg.model).to(device)

    # Linear probing (freeze backbone and train head only)
    if getattr(cfg.model, "linear_probe", False):
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "fc"):  #ResNet50
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):  #DenseNet/EfficientNet for torchvision
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "head"):  #Swin / timm transformers
            for p in model.head.parameters():
                p.requires_grad = True
        elif hasattr(model, "get_classifier"):  #EfficientNet
            for p in model.get_classifier().parameters():
                p.requires_grad = True

        logger.info("Linear probing enabled: backbone frozen, classifier trainable only")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = total_params - trainable_params_count

    logger.info(
        f"Model initialized on {device} | total_params={total_params:,} | "
        f"trainable_params={trainable_params_count:,} | frozen_params={frozen_params_count:,}")
    pct = 100.0 * trainable_params_count / max(1, total_params)
    logger.info(f"Trainable ratio: {pct:.2f}%")


    # --- Datasets & loaders ---
    train_dataset, val_dataset = build_datasets(cfg)

    num_classes, class_names = infer_class_names_and_num_classes(
        train_dataset,
        fallback_names=getattr(cfg.data, "class_names", None),
        fallback_num_classes=getattr(cfg.model, "num_classes", 3) )
    logger.info(f"Inferred classes: num_classes={num_classes} | class_names={class_names}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers)
    val_loader = build_dataloader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers)

    #images, labels = next(iter(train_loader))
    batch = next(iter(train_loader))
    images, labels = batch[0], batch[1]

    logger.info(
        f"Sanity batch: images dtype={images.dtype}, shape={tuple(images.shape)}; "
        f"labels dtype={labels.dtype}, min={labels.min().item()}, max={labels.max().item()}")

    # --- Optimizer / scheduler / callbacks ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay))
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=1e-6)

    callbacks = [
        GradientClippingCallback(max_norm=1.0),
        LRSchedulerCallback(scheduler)]

    # --- Loss ---
    weights, counts = compute_class_weights_from_dataset(train_dataset, num_classes, device=device)
    logger.info(f"Train class counts: {counts.tolist()} | class weights: {weights.detach().cpu().numpy().round(3)}")
    criterion = build_cross_entropy_with_weights(weights)
    # --- Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        log_metrics_fn=log_metrics,
        num_classes=num_classes,
        class_names=class_names,
        callbacks=callbacks,
        early_stopping=getattr(cfg.training, "early_stopping", True),
        patience=getattr(cfg.training, "patience", 5),
        min_delta=getattr(cfg.training, "min_delta", 0.0),
    )

    trainer.fit(train_loader, val_loader, epochs=cfg.training.epochs)
    logger.info("[OK] Training complete")

    # --- Restore best model from MLflow (preferred), else fallback to in-memory best ---
    restored = False
    if mlflow_enabled and mlflow_run_id:
        try:
            log_best = _cfg_get(_cfg_get(_cfg_get(mlflow_cfg, "logging", {}), "log_best", {}), "by", "macro_f1")
            artifact_map = _cfg_get(_cfg_get(_cfg_get(mlflow_cfg, "logging", {}), "log_best", {}), "artifact_path", {})
            if isinstance(artifact_map, dict):
                artifact_path = artifact_map.get(log_best, "best_model_macro_f1" if log_best == "macro_f1" else "best_model")
            else:
                artifact_path = getattr(artifact_map, log_best, "best_model_macro_f1" if log_best == "macro_f1" else "best_model")
            trainer.model = mlflow.pytorch.load_model(f"runs:/{mlflow_run_id}/{artifact_path}").to(device)
            logger.info(f"Restored best model from MLflow: {artifact_path}")
            restored = True
        except Exception as e:
            logger.warning(f"Could not restore best model from MLflow; using in-memory best if available. Error: {e}")

    if not restored:
        trainer.restore_best(which="macro_f1")

    # --- Final evals ---
    _ = trainer.evaluate(val_loader, split="val")
    logger.info("[OK] Evaluation complete")

    test_dataset = build_test_dataset(cfg)
    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers)
    _ = trainer.evaluate(test_loader, split="test", log_confusion_matrix=True)
    logger.info("[OK] Testing complete")

    close_logging()


if __name__ == "__main__":
    main()
