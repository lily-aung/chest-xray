import torch
import mlflow
import wandb
from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logging, log_metrics, close_logging
from src.utils.mlflow_utils import init_mlflow
from src.utils.wandb_utils import init_wandb
from src.data.dataset import build_datasets
from src.data.dataloaders import build_dataloader
from src.models.build_model import build_model
from src.engine.trainer import Trainer

def main():
    cfg = load_config("configs/cnn.yaml"); 
    set_seed(cfg.training.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(cfg)
    run_name = cfg.experiment.run_name or f"seed-{cfg.training.seed}"
    # MLflow init 
    mlflow_run_id = None
    if cfg.logging.use_mlflow:
        mlflow_run_id = init_mlflow(cfg.logging.mlflow_experiment_name, run_name, params={
            "batch_size": cfg.training.batch_size, "lr": cfg.training.lr, "epochs": cfg.training.epochs,
            "seed": cfg.training.seed, "custom_clahe": cfg.augmentation.custom_clahe,
            "horizontal_flip": cfg.augmentation.horizontal_flip, "num_classes": getattr(cfg.model, "num_classes", 3)
        })
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
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized on {device} | trainable_params={num_params:,}")
    # Datasets & loaders
    #train_tfms = train_transforms_with_clahe(cfg.data.img_size) if cfg.augmentation.custom_clahe else train_transforms(cfg.data.img_size)
    #datasets = {
    #    "train": ChestXrayDataset(csv_file=cfg.data.train_csv, transforms=train_tfms),
    #    "val": ChestXrayDataset(csv_file=cfg.data.val_csv, transforms=val_transforms(cfg.data.img_size)),
    #} 
    train_dataset, val_dataset = build_datasets(cfg)
    #train_loader = build_dataloader(datasets["train"], cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    #val_loader = build_dataloader(datasets["val"], cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    train_loader = build_dataloader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    images, labels = next(iter(train_loader))
    logger.info(f"Sanity batch: images dtype={images.dtype}, shape={tuple(images.shape)}; labels dtype={labels.dtype}, min={labels.min().item()}, max={labels.max().item()}")
    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    criterion = torch.nn.CrossEntropyLoss()
    # Trainer
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, logger=logger, log_metrics_fn=log_metrics)
    trainer.fit(train_loader, val_loader, epochs=cfg.training.epochs)

    close_logging(); 
    logger.info("✅Training complete")


if __name__ == "__main__":
    main()
