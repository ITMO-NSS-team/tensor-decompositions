from dataclasses import fields
from .config import TensorGRaDConfig, DataConfig, OptimizerConfig
from .setup_optimizer import setup_optimizer_and_scheduler


def prepared(model, svd_type, rank, scheduler, learning_rate, defaults, optimizer_type, kwargs):
    ranks = rank if isinstance(rank, (tuple, list)) else (rank,)
    if not ranks or len(ranks) > 2:
        raise ValueError("preset rank must be a scalar or a pair of branch ranks")
    parameters = defaults | dict(svd_type=svd_type, rank=ranks[0], second_rank=ranks[-1],
        scheduler=scheduler, learning_rate=learning_rate, optimizer_type=optimizer_type) | kwargs
    data_names = {field.name for field in fields(DataConfig)}
    opt_names = {field.name for field in fields(OptimizerConfig)}
    unknown = parameters.keys() - data_names - opt_names
    if unknown:
        raise ValueError(f"unknown preset options: {sorted(unknown)}")
    config = TensorGRaDConfig(DataConfig(**{k: v for k, v in parameters.items() if k in data_names}),
        OptimizerConfig(**{k: v for k, v in parameters.items() if k in opt_names}))
    return setup_optimizer_and_scheduler(config, model)


class ParallelTG:
    """Factory returning (TensorGRaD, scheduler) with parallel low-rank + sparse branches."""
    _defaults = dict(proj_type="low_rank", galore_2d_proj_type="left", second_proj_type="unstructured_sparse")

    def __new__(cls, model, svd_type, rank, *, scheduler="StepLR", learning_rate=1e-4, **kwargs):
        return prepared(model, svd_type, rank, scheduler, learning_rate, cls._defaults, "tensorgrad_sum", kwargs)


class ULTG:
    """Factory returning (TensorGRaD, scheduler) with sparse + residual low-rank branches."""
    _defaults = dict(proj_type="unstructured_sparse", galore_2d_proj_type="left", second_proj_type="low_rank")

    def __new__(cls, model, svd_type, rank, *, scheduler="StepLR", learning_rate=1e-4, **kwargs):
        return prepared(model, svd_type, rank, scheduler, learning_rate, cls._defaults, "tensorgrad", kwargs)
