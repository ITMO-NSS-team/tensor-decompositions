import torch

SCHEDULER_NAMES = {"ReduceLROnPlateau", "CosineAnnealingLR", "cosine", "StepLR", "step", "constant", "exponential"}


def get_scheduler(scheduler_name, optimizer, gamma=0.1, patience=5, T_max=100, step_size=30):
    """Every declared learning-rate schedule implements step/state_dict/load_state_dict."""
    from ..config import positive_integer
    if scheduler_name not in SCHEDULER_NAMES:
        raise ValueError(f"unknown scheduler={scheduler_name!r}")
    positive_integer("T_max", T_max)
    positive_integer("step_size", step_size)
    if gamma <= 0 or patience < 0:
        raise ValueError("gamma must be positive and patience nonnegative")
    if scheduler_name == "ReduceLROnPlateau":
        if gamma >= 1:
            raise ValueError("ReduceLROnPlateau gamma must be less than 1")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
    if scheduler_name in {"CosineAnnealingLR", "cosine"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    if scheduler_name in {"StepLR", "step"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
