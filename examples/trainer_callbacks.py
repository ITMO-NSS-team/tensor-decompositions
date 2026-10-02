"""Optional Transformers/MLflow adapters for the PR #35 measurement helpers."""
import math

import torch
try:
    from transformers import TrainerCallback
except ImportError as exc:
    raise ImportError("Trainer callbacks require the tdecomp[training] extra") from exc

from .experiment_utils import causal_lm_perplexity, find_projection_scheduler, memory_metrics


def active_logger(logger):
    if logger is None:
        try:
            import mlflow
        except ImportError:
            return None
        logger = mlflow
    return logger if logger.active_run() is not None else None


class UpdateGapMLflowCallback(TrainerCallback):
    def __init__(self, logger=None):
        self.logger = logger
        self._prev_next_update = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._prev_next_update = None

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        scheduler = find_projection_scheduler(optimizer)
        if scheduler is None:
            return
        current = int(scheduler.next_update)
        changed = int(self._prev_next_update is not None and current != self._prev_next_update)
        logger = active_logger(self.logger)
        if logger is not None:
            logger.log_metrics({"ug_next_update": current, "ug_next_update_changed": changed},
                               step=int(state.global_step))
        self._prev_next_update = current


class SystemMetricsCallback(TrainerCallback):
    def __init__(self, logger=None):
        self.logger = logger

    def on_step_end(self, args, state, control, model=None, **kwargs):
        logger = active_logger(self.logger)
        if logger is None:
            return
        import psutil
        metrics = {"cpu_percent": psutil.cpu_percent()}
        device = next((p.device for p in model.parameters() if p.is_cuda), None) if model is not None else None
        if device is not None:
            metrics.update(gpu_memory_used_mb=torch.cuda.memory_allocated(device) / 1e6,
                           gpu_memory_reserved_mb=torch.cuda.memory_reserved(device) / 1e6)
        logger.log_metrics(metrics, step=int(state.global_step))


class PerplexityCallback(TrainerCallback):
    def __init__(self, dataloader=None, max_batches=2, eval_every_n_steps=5,
                 log_key="perplexity", logger=None):
        for name, value in (("max_batches", max_batches), ("eval_every_n_steps", eval_every_n_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.dataloader, self.max_batches = dataloader, max_batches
        self.eval_every_n_steps, self.log_key, self.logger = eval_every_n_steps, log_key, logger

    def on_step_end(self, args, state, control, model=None, eval_dataloader=None,
                    train_dataloader=None, **kwargs):
        logger = active_logger(self.logger)
        if logger is None or model is None or state.global_step % self.eval_every_n_steps:
            return
        dataloader = self.dataloader
        if dataloader is None:
            dataloader = eval_dataloader if eval_dataloader is not None else train_dataloader
        if dataloader is None:
            return
        value = causal_lm_perplexity(model, dataloader, self.max_batches)
        if value is not None and math.isfinite(value):
            logger.log_metrics({self.log_key: value}, step=int(state.global_step))


class PreciseMemoryCallback(TrainerCallback):
    def __init__(self, prefix="mem", logger=None):
        self.prefix, self.logger = prefix, logger

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            device = next((p.device for p in model.parameters() if p.is_cuda), None)
            if device is not None:
                torch.cuda.reset_peak_memory_stats(device)

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        logger = active_logger(self.logger)
        if logger is not None and model is not None and optimizer is not None:
            logger.log_metrics(memory_metrics(model, optimizer, self.prefix), step=int(state.global_step))
