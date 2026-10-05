from dataclasses import dataclass
from typing import Optional, Literal, TypeAlias
from .projectors._common import validate_rank, validate_ratio, sparse_name, svd_name

Galore2DProjectionSide: TypeAlias = Literal["right", "left", "full"]
SparseType: TypeAlias = Literal["topk", "randk", "randomk", "probability"]


def positive_integer(name, value, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class DataConfig:
    batch_size: Optional[int] = None
    n_train: Optional[int] = None
    tmp_dir: str = "/tmp/t2t_datagen"

    def __post_init__(self):
        positive_integer("batch_size", self.batch_size, optional=True)
        positive_integer("n_train", self.n_train, optional=True)


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    optimizer_type: str = "tensorgrad"
    n_epochs: int = 100
    scheduler: str = "cosine"
    gamma: float = 0.1
    scheduler_patience: int = 5
    scheduler_T_max: int = 100
    step_size: int = 30
    rank: object = 128
    scale: float = 1.0
    proj_type: str = "low_rank"
    galore_2d_proj_type: str = "left"
    sparse_ratio: object = 0.1
    sparse_type: str = "topk"
    scale_by_mask_ratio: bool = True
    scaling: Optional[str] = None
    reset_sparse_optimizer_states: bool = True
    moment_policy: str = "reset"
    enforce_full_complex_precision: bool = False
    svd_type: object = "truncated_svd"
    second_proj_type: str = "unstructured_sparse"
    second_sparse_ratio: object = 0.25
    second_sparse_type: str = "topk"
    second_scale: float = 1.0
    second_rank: object = 128
    second_scale_by_mask_ratio: bool = False
    second_scaling: Optional[str] = None
    projection_mode: str = "composite"
    update_proj_gap: int = 100
    update_proj_gap_end: int = 1000
    update_proj_gap_mode: str = "fixed"
    projection_total_iters: Optional[int] = None
    n_iter_max_tucker: int = 10
    tucker_warm_restart: bool = True
    tensorgrad_sum_lambda_sparse: float = 0.05
    tensor_network_type: Optional[str] = None
    tensor_network_chi: Optional[int] = None
    naive_galore: bool = False
    adamw_support_complex: bool = True
    first_dim_rollup: bool = False
    use_checkpoint: bool = False
    cuda: Optional[int] = None
    exclude_first_parameter: bool = False
    random_state: int = 0
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.0

    def __post_init__(self):
        import math
        from .projectors.projector_utils import normalize_group
        if self.optimizer_type not in {"tensorgrad", "tensorgrad_sum", "adamw", "sgd"}:
            raise ValueError(f"unsupported optimizer_type={self.optimizer_type!r}")
        for name in ("n_epochs", "scheduler_T_max", "step_size", "n_iter_max_tucker"):
            positive_integer(name, getattr(self, name))
        if self.projection_total_iters is not None:
            positive_integer("projection_total_iters", self.projection_total_iters)
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be nonnegative")
        if self.scheduler not in {"cosine", "exponential", "StepLR", "step", "constant", "ReduceLROnPlateau", "CosineAnnealingLR"}:
            raise ValueError(f"unknown scheduler={self.scheduler!r}")
        for name in ("learning_rate", "weight_decay", "eps", "momentum", "tensorgrad_sum_lambda_sparse"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.gamma, (int, float)) or not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if self.scheduler == "ReduceLROnPlateau" and self.gamma >= 1:
            raise ValueError("ReduceLROnPlateau gamma must be less than 1")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer seed")
        for name in ("scale_by_mask_ratio", "second_scale_by_mask_ratio", "reset_sparse_optimizer_states",
                     "enforce_full_complex_precision", "tucker_warm_restart", "naive_galore",
                     "adamw_support_complex", "first_dim_rollup", "use_checkpoint", "exclude_first_parameter"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        normalized = normalize_group(vars(self) | {"epochs": self.n_epochs})
        self.svd_type = normalized["svd_type"]
        self.sparse_type, self.second_sparse_type = normalized["sparse_type"], normalized["second_sparse_type"]


@dataclass
class WandBConfig:
    log_ranks_interval: int = 100
    log_gradients: bool = False
    log_projections: bool = False


@dataclass
class TensorGRaDConfig:
    data: DataConfig
    opt: OptimizerConfig
    model_type: str = "neural_network"
