import os
import torch

class GradientClippingCallback:
    def __init__(self, max_norm=1.0):
        self.max_norm = float(max_norm)

    def on_batch_end(self, trainer, epoch: int, batch: int):
        params = [p for p in trainer.model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.max_norm)

class LRSchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._last_epoch_stepped = None

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        if self._last_epoch_stepped == epoch:
            return
        self.scheduler.step()
        self._last_epoch_stepped = epoch
        metrics["lr"] = float(trainer.optimizer.param_groups[0]["lr"])


class ModelCheckpointCallback:
    def __init__(self, dirpath="checkpoints", monitor="val_macro_f1", mode="max", save_last=False, fname="best.pt"):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_last = bool(save_last)
        self.fname = fname
        self.best = None

    def _is_better(self, value: float):
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        os.makedirs(self.dirpath, exist_ok=True)

        state_cpu = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.pt")
            torch.save({"epoch": epoch, "state_dict": state_cpu}, last_path)

        if self.monitor not in metrics:
            return

        value = float(metrics[self.monitor])
        if not self._is_better(value):
            return

        self.best = value
        best_path = os.path.join(self.dirpath, self.fname)

        torch.save(
            {"epoch": epoch, "monitor": self.monitor, "best": self.best, "state_dict": state_cpu},
            best_path
        )

        if self.monitor == "val_macro_f1":
            trainer.best_state_dict_macro_f1 = {k: v.clone() for k, v in state_cpu.items()}
        elif self.monitor == "val_accuracy":
            trainer.best_state_dict_acc = {k: v.clone() for k, v in state_cpu.items()}

        if trainer.logger:
            trainer.logger.info(f"[Checkpoint] New best {self.monitor}={value:.4f} saved to {best_path}")


def wrap_log_metrics_fn(base_log_metrics_fn, trainer):
    def _wrapped(metrics: dict, epoch: int):
        for cb in getattr(trainer, "callbacks", []):
            fn = getattr(cb, "on_metrics", None)
            if callable(fn):
                fn(trainer, metrics, epoch)
        return base_log_metrics_fn(metrics, epoch)
    return _wrapped
