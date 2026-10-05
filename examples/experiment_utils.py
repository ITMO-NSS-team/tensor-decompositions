"""Portable measurement helpers adapted from PR #35.

Trainer callbacks are loaded only when requested. Memory values count tensor
storage; allocator peaks include other allocations and are reported separately.
"""
import itertools
import math

import torch
import torch.nn.functional as F

_CALLBACKS = {"UpdateGapMLflowCallback", "SystemMetricsCallback", "PerplexityCallback", "PreciseMemoryCallback"}


def __getattr__(name):
    if name in _CALLBACKS:
        from . import trainer_callbacks
        return getattr(trainer_callbacks, name)
    raise AttributeError(name)


def find_projection_scheduler(optimizer):
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        for state in getattr(optimizer, "state", {}).values():
            for name in ("first_proj", "second_proj"):
                scheduler = getattr(state.get(name), "update_gap_scheduler", None)
                if scheduler is not None:
                    return scheduler
        optimizer = getattr(optimizer, "optimizer", None)
    return None


def causal_lm_perplexity(model, dataloader, max_batches=2):
    """Evaluate shifted causal labels, excluding padding and ignore_index=-100."""
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0:
        raise ValueError("max_batches must be a positive integer")
    device = next(model.parameters(), torch.empty(0)).device
    was_training = model.training
    module_modes = [(module, module.training) for module in model.modules()]
    total_nll, total_tokens = 0.0, 0
    try:
        model.eval()
        with torch.no_grad():
            for batch in itertools.islice(dataloader, max_batches):
                inputs = {name: value.to(device) if torch.is_tensor(value) else value
                          for name, value in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                logits, labels = logits[..., :-1, :], inputs["labels"][..., 1:]
                valid = labels != -100
                if "attention_mask" in inputs:
                    valid = valid & inputs["attention_mask"][..., 1:].bool()
                count = int(valid.sum())
                if count:
                    total_nll += F.cross_entropy(logits[valid], labels[valid], reduction="sum").item()
                    total_tokens += count
    finally:
        model.train(was_training)
        for module, mode in module_modes:
            module.training = mode
    if total_tokens == 0:
        return None
    try:
        return math.exp(total_nll / total_tokens)
    except OverflowError:
        return math.inf


def tensor_storage_bytes(value):
    """Count unique tensor storage, including tensor data inside projectors."""
    visited, storages = set(), {}

    def visit(item):
        if id(item) in visited:
            return
        visited.add(id(item))
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            storages[key] = (storage.nbytes(), item.is_cuda)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "_data_fields"):
            visit(vars(item))

    visit(value)
    return sum(size for size, _ in storages.values()), sum(size for size, cuda in storages.values() if cuda)


def memory_metrics(model, optimizer, prefix="mem"):
    weights, weight_cuda = tensor_storage_bytes(list(model.parameters()))
    gradients, grad_cuda = tensor_storage_bytes([p.grad for p in model.parameters() if p.grad is not None])
    while not hasattr(optimizer, "state") and hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    state, state_cuda = tensor_storage_bytes(dict(optimizer.state))
    metrics = {f"{prefix}_weights_mb": weights / 1024**2,
               f"{prefix}_grads_mb": gradients / 1024**2,
               f"{prefix}_opt_state_mb": state / 1024**2}
    device = next((p.device for p in model.parameters() if p.is_cuda), None)
    if device is not None:
        allocated, peak = torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)
        metrics.update({f"{prefix}_cuda_allocated_mb": allocated / 1024**2,
                        f"{prefix}_cuda_peak_allocated_mb": peak / 1024**2,
                        f"{prefix}_other_cuda_allocations_estimate_mb":
                            max(0, allocated - weight_cuda - grad_cuda - state_cuda) / 1024**2})
    return metrics


def print_param_shapes(optimizer, only_group_index=None):
    for index, group in enumerate(optimizer.param_groups):
        if only_group_index is None or index == only_group_index:
            print("group", index, [tuple(parameter.shape) for parameter in group["params"]])


def get_batch_size_mb(batch_encoding):
    return tensor_storage_bytes(batch_encoding)[0] / 1024**2


def get_model_size_mb(model):
    return tensor_storage_bytes(list(model.parameters()))[0] / 1024**2
