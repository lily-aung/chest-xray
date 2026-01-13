import torch
from tqdm import tqdm
import mlflow, wandb

class Trainer:
    def __init__(self, model, optimizer, criterion, device, logger=None, log_metrics_fn=None, best_val_acc=0.0):
        self.model = model; self.optimizer = optimizer; self.criterion = criterion; self.device = device
        self.logger = logger; self.log_metrics_fn = log_metrics_fn; self.best_val_acc = best_val_acc

    def _prep_batch(self, images, labels):
        images = images.to(self.device); labels = labels.to(self.device).long()
        if images.dtype != torch.float32 and images.dtype != torch.float16: images = images.float()
        if images.max() > 1.5: images = images / 255.0
        return images, labels

    def _run_epoch(self, dataloader, epoch, is_training=True):
        self.model.train() if is_training else self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        phase = "Train" if is_training else "Validation"
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} [{phase}]"):
            images, labels = self._prep_batch(images, labels)
            if is_training: self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_training):
                logits = self.model(images); loss = self.criterion(logits, labels)
                if is_training: loss.backward(); self.optimizer.step()
            running_loss += loss.item()
            preds = logits.argmax(dim=1); correct += (preds == labels).sum().item(); total += labels.size(0)
        return running_loss / max(1, len(dataloader)), correct / max(1, total)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, epoch, True)
            val_loss, val_acc = self._run_epoch(val_loader, epoch, False)
            metrics = {"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}
            if self.log_metrics_fn: self.log_metrics_fn(metrics, epoch)
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if mlflow.active_run() and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                mlflow.log_metric("best_val_accuracy", val_acc)
                mlflow.pytorch.log_model(self.model, artifact_path="best_model")
                if wandb.run:
                    wandb.run.summary["best_val_accuracy"] = float(val_acc)
                if self.logger: self.logger.info(f"New best model logged to MLflow (val_acc={val_acc:.4f})")
