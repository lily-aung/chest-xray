import wandb

def init_wandb(project, run_name, config, mlflow_run_id=None):
    wandb.init( project=project, name=run_name, config=config)
    if mlflow_run_id:
        wandb.config.update( {"mlflow_run_id": mlflow_run_id},allow_val_change=True)

def log_metrics(metrics: dict, step: int):
    """
    Log metrics to WandB safely.
    """
    if wandb.run is None:
        return
    payload = dict(metrics)
    payload["epoch"] = step
    wandb.log(payload)


def log_images(images: dict):
    """
    Log images to WandB.
    """
    if wandb.run is None:
        return
    wandb.log(images)


def close_wandb():
    """
    Finish WandB run safely.
    """
    if wandb.run is not None:
        wandb.finish()

