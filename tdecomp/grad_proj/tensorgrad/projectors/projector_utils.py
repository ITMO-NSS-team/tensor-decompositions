"""Explicit single/composite projector factory and portable-state registry."""
import math
from .galore_projector import GaLoreProjector
from .tensor_lowrank_projector import TensorGradLowRankProjector
from .tensor_sparse_projector import TensorGradSparseProjector
from .tensor_unstructured_sparse_projector import TensorGradUnstructuredProjector
from .sparse_projector import GaLoreSparseProjector
from .update_gap_scheduler import UpdateGapScheduler
from ._common import svd_name, validate_rank, validate_ratio, sparse_name

PROJECTOR_TYPES = {"low_rank", "structured_sparse", "unstructured_sparse"}
PROJECTOR_REGISTRY = {cls.__name__: cls for cls in (
    GaLoreProjector, TensorGradLowRankProjector, TensorGradSparseProjector,
    TensorGradUnstructuredProjector, GaLoreSparseProjector)}


def normalize_group(group):
    """Validate a copy before allocating states or changing caller-owned groups."""
    result = dict(group)
    defaults = dict(proj_type="low_rank", second_proj_type="unstructured_sparse",
        second_rank=group.get("rank", 128), scale=1.0, second_scale=1.0,
        sparse_ratio=0.1, second_sparse_ratio=0.25, sparse_type="topk",
        second_sparse_type="topk", scale_by_mask_ratio=False,
        second_scale_by_mask_ratio=False, svd_type="truncated_svd",
        galore_2d_proj_type="left", projection_mode="composite",
        update_proj_gap=100, update_proj_gap_end=1000, update_proj_gap_mode="fixed",
        n_iter_max_tucker=10, tucker_warm_restart=False, random_state=0,
        moment_policy="reset", lambda_sparse=1.0)
    for key, value in defaults.items():
        result.setdefault(key, value)
    if result["projection_mode"] not in {"single", "composite"}:
        raise ValueError("projection_mode must be single or composite")
    if result["moment_policy"] != "reset":
        raise ValueError("moment_policy currently supports reset only")
    if not isinstance(result["lambda_sparse"], (int, float)) or not math.isfinite(result["lambda_sparse"]) or result["lambda_sparse"] < 0:
        raise ValueError("lambda_sparse must be finite and nonnegative")
    if "reset_sparse_optimizer_states" in group and group["reset_sparse_optimizer_states"] is not True:
        raise ValueError("reset_sparse_optimizer_states=False is incompatible with moment_policy='reset'")
    for prefix in ("", "second_") if result["projection_mode"] == "composite" else ("",):
        if result[prefix + "proj_type"] not in PROJECTOR_TYPES:
            raise ValueError(f"unknown {prefix}proj_type")
        validate_rank(result[prefix + "rank"])
        validate_ratio(result[prefix + "sparse_ratio"])
        result[prefix + "sparse_type"] = sparse_name(result[prefix + "sparse_type"])
        scaling = result.get(prefix + "scaling")
        if scaling is not None and scaling not in {"none", "energy", "unbiased"}:
            raise ValueError("scaling must be none, energy or unbiased")
        if scaling == "unbiased" and result[prefix + "sparse_type"] != "randk":
            raise ValueError("unbiased scaling requires uniform randk sampling")
    result["svd_type"] = svd_name(result["svd_type"])
    if result["galore_2d_proj_type"] not in {"left", "right", "full"}:
        raise ValueError("galore_2d_proj_type must be left, right or full")
    # Constructing these validates every shape-independent argument immediately.
    get_projector(result, validate_only=True)
    return result


def get_projector(group, matrix_only=False, support_complex=False, validate_only=False):
    scheduler = UpdateGapScheduler(group.get("update_proj_gap", 100), group.get("update_proj_gap_end", 1000),
        group.get("update_proj_gap_mode", "fixed"), group.get("batch_size"), group.get("epochs", 1),
        group.get("training_samples"), total_iters=group.get("projection_total_iters", group.get("scheduler_T_max", 100)))
    mode = group.get("projection_mode", "composite" if group.get("optimizer_type", "tensorgrad").startswith("tensorgrad") else "single")
    first = create_projector(group.get("proj_type", "low_rank"), group, scheduler,
                             matrix_only, support_complex)
    if mode == "single":
        return first
    if mode != "composite":
        raise ValueError("projection_mode must be single or composite")
    second_scheduler = UpdateGapScheduler(scheduler.update_gap, scheduler.update_gap_end,
        scheduler.mode, total_iters=scheduler.total_iters)
    second = create_projector(group.get("second_proj_type", "unstructured_sparse"), group,
        second_scheduler, matrix_only, support_complex, prefix="second_")
    return first, second


def _get_param(group, name, prefix=""):
    return group.get(prefix + name, group.get(name))


def create_projector(proj_type, group, update_gap_scheduler, matrix_only=False, support_complex=False, prefix=""):
    if proj_type not in PROJECTOR_TYPES:
        raise ValueError(f"Unknown projector type={proj_type!r}")
    scale = _get_param(group, "scale", prefix)
    seed = group.get("random_state", 0) + (1 if prefix else 0)
    kwargs = dict(update_gap_scheduler=update_gap_scheduler, scale=1.0 if scale is None else scale, random_state=seed)
    if proj_type != "low_rank":
        kwargs.update(sparse_ratio=_get_param(group, "sparse_ratio", prefix) or 0.25,
            sparse_type=_get_param(group, "sparse_type", prefix) or "topk",
            scale_by_mask_ratio=_get_param(group, "scale_by_mask_ratio", prefix) or False,
            scaling=group.get(prefix + "scaling"))
        cls = TensorGradUnstructuredProjector if proj_type == "unstructured_sparse" else TensorGradSparseProjector
        return cls(**kwargs)
    kwargs.update(rank=_get_param(group, "rank", prefix), svd_type=svd_name(group.get("svd_type")))
    if group.get("dim", 2) <= 2 or matrix_only:
        return GaLoreProjector(**kwargs, galore_2d_proj_type=group.get("galore_2d_proj_type", "left"),
            support_complex=support_complex, activation_checkpoint=group.get("use_checkpoint", False))
    return TensorGradLowRankProjector(**kwargs, warm_restart=group.get("tucker_warm_restart", False),
        n_iter_max=group.get("n_iter_max_tucker", 10))


def projector_from_state(state, parameter):
    if not isinstance(state, dict) or state.get("version") != 1 or state.get("kind") not in PROJECTOR_REGISTRY:
        raise ValueError("unsupported or invalid projector state")
    try:
        projector = PROJECTOR_REGISTRY[state["kind"]](**state["config"])
        projector.load_state_dict(state, parameter.device, parameter.dtype)
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete projector state") from exc
    return projector
