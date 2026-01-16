import torch
import mlflow
import wandb
from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logging, log_metrics, close_logging
from src.utils.mlflow_utils import init_mlflow
from src.utils.wandb_utils import init_wandb
from src.data.dataset import build_datasets, build_test_dataset
from src.data.dataloaders import build_dataloader
from src.models.build_model import build_model
from src.engine.trainer import Trainer
from src.utils.losses import build_cross_entropy_with_weights, compute_class_weights_from_dataset
import torch.optim.lr_scheduler as lr_scheduler
from src.engine.callbacks import GradientClippingCallback, LRSchedulerCallback, wrap_log_metrics_fn
from src.data.dataset import (build_datasets, build_test_dataset, infer_class_names_and_num_classes)

def main():
    cfg = load_config("configs/cnn.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml").mlflow

    set_seed(cfg.training.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(cfg)
    run_name = cfg.experiment.run_name or f"seed-{cfg.training.seed}"

    # MLflow init 
    mlflow_run_id = None
    if cfg.logging.use_mlflow:
        mlflow_run_id = init_mlflow(
            cfg.logging.mlflow_experiment_name, run_name, params={
                "batch_size": cfg.training.batch_size, "lr": cfg.training.lr, "epochs": cfg.training.epochs,
                "seed": cfg.training.seed, "custom_clahe": cfg.augmentation.custom_clahe,
                "horizontal_flip": cfg.augmentation.horizontal_flip, "num_classes": getattr(cfg.model, "num_classes", 3),
                "linear_probe": getattr(cfg.model, "linear_probe", False)
            }
        )
        mlflow.pytorch.autolog(log_models=False)
        logger.info(f"[MLflow] run_id={mlflow_run_id}")

    # W&B init 
    if cfg.logging.use_wandb: 
        init_wandb(cfg.logging.wandb_project, run_name, config=cfg, mlflow_run_id=mlflow_run_id)

    # Sync IDs (MLflow ←→ W&B)
    if mlflow.active_run() and wandb.run:
        mlflow.set_tag("wandb_run_id", wandb.run.id)
        mlflow.set_tag("wandb_url", wandb.run.url)

    # Model
    model = build_model(cfg.model).to(device)

    # Linear probing (freeze backbone, train head only)
    if getattr(cfg.model, "linear_probe", False):
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze classifier / head by model type
        if hasattr(model, "fc"):                         # ResNet
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):               # DenseNet / EfficientNet (torchvision)
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "head"):                     # Swin / timm transformers
            for p in model.head.parameters():
                p.requires_grad = True
        elif hasattr(model, "get_classifier"):           # timm EfficientNet
            for p in model.get_classifier().parameters():
                p.requires_grad = True

        logger.info("Linear probing enabled: backbone frozen, classifier trainable only")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized on {device} | trainable_params={num_params:,}")

    # Datasets & loaders
    #train_tfms = train_transforms_with_clahe(cfg.data.img_size) if cfg.augmentation.custom_clahe else train_transforms(cfg.data.img_size)
    #datasets = {
    #    "train": ChestXrayDataset(csv_file=cfg.data.train_csv, transforms=train_tfms),
    #    "val": ChestXrayDataset(csv_file=cfg.data.val_csv, transforms=val_transforms(cfg.data.img_size)),
    #} 

    train_dataset, val_dataset = build_datasets(cfg)
    num_classes, class_names = infer_class_names_and_num_classes(
        train_dataset,  fallback_names=getattr(cfg.data, "class_names", None),
        fallback_num_classes=getattr(cfg.model, "num_classes", 3))
    logger.info(f"Inferred classes: num_classes={num_classes} | class_names={class_names}")

    train_loader = build_dataloader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    images, labels = next(iter(train_loader))
    logger.info(
        f"Sanity batch: images dtype={images.dtype}, shape={tuple(images.shape)}; "
        f"labels dtype={labels.dtype}, min={labels.min().item()}, max={labels.max().item()}" )

    # Optimizer & loss (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    # Scheduler: cosine annealing over the full training run
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs), eta_min=1e-6)
    callbacks = [ GradientClippingCallback(max_norm=1.0), LRSchedulerCallback(scheduler),
    
    #criterion = torch.nn.CrossEntropyLoss()
    weights, counts = compute_class_weights_from_dataset(train_dataset, num_classes , device=device)
    logger.info(f"Train class counts: {counts.tolist()} | class weights: {weights.detach().cpu().numpy().round(3)}")
    criterion = build_cross_entropy_with_weights(weights)

    # Trainer
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion,
        device=device, logger=logger, log_metrics_fn=None, 
        num_classes=num_classes, class_names=class_names, callbacks=callbacks )
    trainer.log_metrics_fn = wrap_log_metrics_fn(log_metrics, trainer)

    trainer.fit(train_loader, val_loader, epochs=cfg.training.epochs)
    
    logger.info("[OK]Training complete")
    # Evaluate best model on validation or test

    if mlflow_cfg.enabled and mlflow_run_id:
        best_key = mlflow_cfg.logging.log_best.by
        artifact_path = mlflow_cfg.logging.log_best.artifact_path[best_key]

        trainer.model = mlflow.pytorch.load_model(
            f"runs:/{mlflow_run_id}/{artifact_path}"
        ).to(device)

        logger.info(f"Restored best model from MLflow: {artifact_path}")
    else:     trainer.restore_best(which="macro_f1")#"accuracy"

    results = trainer.evaluate(val_loader, split="val")
    logger.info("[OK]Evaluation complete")
    # Evaluate
    test_dataset = build_test_dataset(cfg)
    test_loader = build_dataloader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)    
    test_results = trainer.evaluate(test_loader, split="test", log_confusion_matrix=True)
    logger.info("[OK]Testing complete")
    close_logging()

if __name__ == "__main__":
    main()
