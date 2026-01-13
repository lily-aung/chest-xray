### ML Flow
import mlflow
mlflow.set_experiment("debug-mlflow")
with mlflow.start_run():
    mlflow.log_metric("test_metric", 1.0)

### WAN Flow
import wandb
wandb.init( project="chest-xray",name="local-dev-run" )
wandb.log({ "loss": 0.42, "accuracy": 0.88 })
wandb.finish()
