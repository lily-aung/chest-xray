import torch
from tqdm import tqdm
import mlflow, wandb
from src.utils.metrics import confusion_matrix_torch, per_class_precision_recall_f1
import os, tempfile
from src.utils.mlflow_utils import save_confusion_matrix_png

class Trainer:
    def __init__(
        self, model, optimizer, criterion, device, logger=None, log_metrics_fn=None, best_val_acc=0.0,
        num_classes=3, class_names=None, early_stopping=False, patience=5, min_delta=0.0, callbacks=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.log_metrics_fn = log_metrics_fn

        self.best_val_acc = float(best_val_acc)
        self.num_classes = int(num_classes)
        self.class_names = class_names or [str(i) for i in range(self.num_classes)]

        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_val_macro_f1 = -1.0
        self.best_val_macro_f1_epoch = -1
        self.es_bad_epochs = 0

        self.best_state_dict_acc = None
        self.best_state_dict_macro_f1 = None

        self.callbacks = callbacks or []

    def _cb(self, hook, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(self, *args, **kwargs)

    def _prep_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device).long()
        if images.dtype not in (torch.float32, torch.float16):
            images = images.float()
        return images, labels

    def _run_epoch(self, dataloader, epoch, is_training=True):
        self.model.train() if is_training else self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        phase = "Train" if is_training else "Validation"

        for batch_idx, (images, labels, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [{phase}]")):
            if epoch == 0 and is_training and batch_idx == 0 and self.logger:
                self.logger.info(" First training batch started ~~~.")

            images, labels = self._prep_batch(images, labels)
            if is_training:
                self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()

                # gradient clipping callback
                #self._cb("on_batch_end", step=(epoch, batch_idx))
                self._cb("on_batch_end", epoch=epoch, batch=batch_idx)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / max(1, len(dataloader)), correct / max(1, total)

    def evaluate(self, dataloader, split="val", step=None, log_confusion_matrix=False):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc=f"[Evaluate:{split}]"):
                images, labels = self._prep_batch(images, labels)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_loss = running_loss / max(1, len(dataloader))
        accuracy = correct / max(1, total)
        y_pred = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        y_true = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)

        cm = confusion_matrix_torch(y_true, y_pred, self.num_classes)
        precision, recall, f1, support = per_class_precision_recall_f1(cm)
        macro_f1 = float(f1.mean().item())
        macro_recall = float(recall.mean().item())
        balanced_accuracy = macro_recall
        metrics = {
            f"{split}_loss": avg_loss,
            f"{split}_accuracy": accuracy,
            f"{split}_macro_f1": macro_f1,
            f"{split}_macro_recall": macro_recall,
            f"{split}_balanced_accuracy": balanced_accuracy}

        for i, name in enumerate(self.class_names):
            metrics[f"{split}_recall_{name}"] = float(recall[i].item())
            metrics[f"{split}_f1_{name}"] = float(f1[i].item())
            metrics[f"{split}_support_{name}"] = float(support[i].item())

        #step_int = int(step) if step is not None else 0
        step_int = int(step) if step is not None else -1

        if self.log_metrics_fn:
            self.log_metrics_fn(metrics, step_int)

        if mlflow.active_run():
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=step_int)

        if mlflow.active_run() and log_confusion_matrix:
            with tempfile.TemporaryDirectory() as td:
                out_path = os.path.join(td, f"{split}_confusion_matrix.png")
                save_confusion_matrix_png(cm, self.class_names, out_path, title=f"{split.upper()} Confusion Matrix")
                mlflow.log_artifact(out_path, artifact_path=f"{split}_artifacts")

        # summary for "final" evals
        if wandb.run and step is None:
            for k, v in metrics.items():
                wandb.run.summary[k] = float(v)

        if self.logger:
            self.logger.info(f"[{split.upper()}] Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

        return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1, "cm": cm, "preds": y_pred, "labels": y_true}


    def fit(self, train_loader, val_loader, epochs):
        #self.preflight_loader(train_loader, n_batches=2, name="train")
        #self.preflight_loader(val_loader, n_batches=1, name="val")
        for epoch in range(epochs):
            self._cb("on_epoch_start", epoch=epoch)
            train_loss, train_acc = self._run_epoch(train_loader, epoch, True)
            val_out = self.evaluate(val_loader, split="val", step=epoch, log_confusion_matrix=False)

            val_loss, val_acc, val_macro_f1 = val_out["loss"], val_out["accuracy"], val_out["macro_f1"]
            cm, y_true, y_pred = val_out["cm"], val_out["labels"], val_out["preds"]

            metrics = { "train_loss": train_loss, "train_accuracy": train_acc,
                "val_loss": val_loss, "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "lr": float(self.optimizer.param_groups[0]["lr"])}

            # allow callbacks (scheduler) to update lr and metrics
            #self._cb("on_epoch_end", epoch=epoch, logs=metrics)
            self._cb("on_epoch_end", epoch=epoch, metrics=metrics)

            if self.log_metrics_fn:
                self.log_metrics_fn(metrics, epoch)

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")
            # best-by-accuracy
            if mlflow.active_run() and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state_dict_acc = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                mlflow.log_metric("best_val_accuracy", float(val_acc), step=epoch)
                mlflow.pytorch.log_model(self.model, artifact_path="best_model")
                self._log_confusion(cm, y_true, y_pred, split="val", step=epoch)
            # best-by-macro-f1 + early stop
            if val_macro_f1 > self.best_val_macro_f1 + self.min_delta:
                self.best_val_macro_f1 = val_macro_f1
                self.best_val_macro_f1_epoch = epoch + 1
                self.es_bad_epochs = 0

                if mlflow.active_run():
                    mlflow.log_metric("best_val_macro_f1", float(val_macro_f1), step=epoch)
                    mlflow.log_metric("best_val_macro_f1_epoch", float(epoch + 1), step=epoch)
                    mlflow.pytorch.log_model(self.model, artifact_path="best_model_macro_f1")

                self.best_state_dict_macro_f1 = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self._log_confusion(cm, y_true, y_pred, split="val", step=epoch)
            else:
                self.es_bad_epochs += 1

            if self.early_stopping and self.es_bad_epochs >= self.patience:
                if self.logger:
                    self.logger.info(
                        f"Early stopping at epoch {epoch+1} | best_val_macro_f1={self.best_val_macro_f1:.4f} "
                        f"(epoch {self.best_val_macro_f1_epoch})" )
                break
    
    def restore_best(self, which="macro_f1"):
        if which == "macro_f1" and self.best_state_dict_macro_f1 is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict_macro_f1.items()})
            if self.logger:
                self.logger.info("Restored best model by macro-F1 (from memory).")
            return True

        if which == "accuracy" and self.best_state_dict_acc is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict_acc.items()})
            if self.logger:
                self.logger.info("Restored best model by accuracy (from memory).")
            return True

        if self.logger:
            self.logger.warning(f"No best checkpoint stored for which='{which}'. Keeping current weights.")
        return False

    def _log_confusion(self, cm, y_true, y_pred, split, step):
        if mlflow.active_run():
            with tempfile.TemporaryDirectory() as td:
                out_path = os.path.join(td, f"{split}_confusion_matrix_step{step}.png")
                save_confusion_matrix_png(
                    cm, self.class_names, out_path,
                    title=f"{split.upper()} Confusion Matrix" )
                mlflow.log_artifact(out_path, artifact_path=f"{split}_artifacts")

    def preflight_loader(self, dataloader, n_batches=2, name="train"):
        import time
        t0 = time.time()
        it = iter(dataloader)
        for i in range(n_batches):
            if self.logger:
                self.logger.info(f"🔎 Preflight {name}: fetching batch {i+1}/{n_batches}...")
            try:
                _ = next(it)
            except Exception as e:
                if self.logger:
                    self.logger.exception(f"Preflight {name} failed on batch {i+1}: {e}")
                raise
        if self.logger:
            self.logger.info(f" Preflight {name} OK: {n_batches} batches in {time.time()-t0:.2f}s")
