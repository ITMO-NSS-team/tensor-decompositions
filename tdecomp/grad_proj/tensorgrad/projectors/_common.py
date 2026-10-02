"""Shared Torch projector contracts: ownership, scaling, RNG and safe state."""
import math
import numbers
import torch

SVD_NAMES = {"truncated_svd", "full_svd", "randomized_svd"}
SPARSE_NAMES = {"topk", "randk", "probability"}


def sparse_name(value):
    value = {"randomk": "randk", "probablility": "probability"}.get(value, value)
    if value not in SPARSE_NAMES:
        raise ValueError(f"Unknown sparse_type={value!r}")
    return value


def validate_rank(rank):
    values = rank if isinstance(rank, (tuple, list)) else [rank]
    if not values:
        raise ValueError("rank must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError("rank must contain positive integers or fractions in (0, 1]")
        if isinstance(value, numbers.Integral):
            if value <= 0:
                raise ValueError("integer rank must be positive")
        elif not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("fractional rank must be in (0, 1]")


def rank_size(rank, size):
    validate_rank(rank)
    return min(size, int(rank)) if isinstance(rank, numbers.Integral) else max(1, math.ceil(rank * size))


def validate_ratio(ratio):
    values = ratio if isinstance(ratio, (tuple, list)) else [ratio]
    if not values or any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or not 0 < x <= 1 for x in values):
        raise ValueError("sparse_ratio must contain fractions in (0, 1]")


def kept_size(ratio, size):
    """Keep ceil(ratio * size), without an extra element on integer boundaries."""
    return min(size, max(1, math.ceil(ratio * size)))


def svd_name(method):
    if method is None:
        return "truncated_svd"
    if isinstance(method, str):
        if method not in SVD_NAMES:
            raise ValueError(f"Unknown SVD method={method!r}")
        return method
    if not callable(method):
        raise ValueError("svd_type must be a known name or callable")
    import tensorly as tl
    for name in SVD_NAMES:
        if method is getattr(tl, name, None) or method is getattr(tl.tenalg.svd, name, None):
            return name
    return method


def write_back(result, output_buffer=None, alpha=1.0, accumulate=False):
    """Return alpha*result; a provided buffer is fully overwritten or added to."""
    if output_buffer is None:
        return result * alpha
    if output_buffer.shape != result.shape or output_buffer.device != result.device or output_buffer.dtype != result.dtype:
        raise ValueError("output_buffer must match reconstructed shape, dtype and device")
    if accumulate:
        output_buffer.add_(result, alpha=alpha)
    else:
        output_buffer.copy_(result * alpha)
    return output_buffer


def tensors_to(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() or value.is_complex() else value.dtype)
    if isinstance(value, (tuple, list)):
        return [tensors_to(x, device, dtype) for x in value]
    if isinstance(value, dict):
        return {k: tensors_to(v, device, dtype) for k, v in value.items()}
    return value


class ProjectorState:
    """Each projector owns a CPU generator; sampling never changes application RNG."""
    _data_fields = ()
    _config_fields = ()

    def _init_state(self, scheduler=None, random_state=0):
        from .update_gap_scheduler import UpdateGapScheduler
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer seed")
        self.update_gap_scheduler = scheduler if scheduler is not None else UpdateGapScheduler(100, 1000)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(random_state)
        self.should_update = False
        self._orig_shape = None

    def _check_input(self, x):
        if not isinstance(x, torch.Tensor) or x.layout != torch.strided or not (x.is_floating_point() or x.is_complex()):
            raise ValueError("projectors require dense floating-point Torch tensors")
        if x.ndim < 2 or x.numel() == 0:
            raise ValueError("projectors require at least two nonempty dimensions")
        if x.dtype not in {torch.float32, torch.float64, torch.complex64, torch.complex128}:
            raise ValueError("projectors support float32, float64, complex64 and complex128")
        if not torch.isfinite(x).all():
            raise ValueError("projectors require finite input values")
        shape = tuple(x.shape)
        if self._orig_shape is not None and self._orig_shape != shape:
            raise ValueError("a projector cannot change its input shape")
        return shape

    def _update_due(self, iteration, initialized):
        # Advance at initialization, without short-circuiting the scheduler.
        self.should_update = self.update_gap_scheduler.should_update(iteration)
        return self.should_update or not initialized

    def _seed(self):
        return int(torch.randint(0, 2**31 - 1, (), generator=self.generator).item())

    def state_dict(self):
        config = {key: getattr(self, key) for key in self._config_fields}
        if callable(config.get("svd_type")):
            raise ValueError("custom SVD callables cannot be saved in the portable state format; use a registered SVD name")
        return {"version": 1, "kind": type(self).__name__, "config": config,
                "data": {key: getattr(self, key) for key in self._data_fields},
                "shape": self._orig_shape, "scheduler": self.update_gap_scheduler.state_dict(),
                "rng": self.generator.get_state()}

    def load_state_dict(self, state, device, dtype):
        if state.get("version") != 1 or state.get("kind") != type(self).__name__:
            raise ValueError("unsupported projector state version or kind")
        if set(state.get("data", {})) != set(self._data_fields):
            raise ValueError("incomplete projector data")
        self.update_gap_scheduler.load_state_dict(state["scheduler"])
        self.generator.set_state(state["rng"].cpu())
        self._orig_shape = tuple(state["shape"]) if state["shape"] is not None else None
        for key, value in state["data"].items():
            setattr(self, key, tensors_to(value, device, dtype))
