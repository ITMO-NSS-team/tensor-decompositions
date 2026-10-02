import math
import torch
from .abstract_sparce_projector import AbstractSparceProjector
from ._common import kept_size, sparse_name, validate_ratio, write_back


class TensorGradUnstructuredProjector(AbstractSparceProjector):
    """Gather flat coordinates. Back projection applies the named mask scaling.

    'none' preserves retained coordinates; 'energy' uses 1/sqrt(p), which preserves
    expected squared norm only for uniform masks; 'unbiased' uses 1/p for uniform
    randk masks. topk and probability offer no unbiasedness guarantee.
    """
    _config_fields = ("sparse_ratio", "sparse_type", "scale", "scale_by_mask_ratio", "scaling", "proj_type")
    _data_fields = ("_indices", "scale_factor")

    def __init__(self, sparse_ratio=0.25, sparse_type="randk", verbose=False,
                 update_gap_scheduler=None, scale=1.0, proj_type="std",
                 warm_restart=False, n_iter_max=10, scale_by_mask_ratio=False,
                 scaling=None, random_state=0):
        validate_ratio(sparse_ratio)
        if isinstance(sparse_ratio, (tuple, list)):
            raise ValueError("unstructured sparse_ratio must be scalar")
        self.sparse_ratio, self.sparse_type = sparse_ratio, sparse_name(sparse_type)
        self.scale = self.base_scale = float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")
        self.verbose, self.proj_type = verbose, proj_type
        self.warm_restart, self.n_iter_max = warm_restart, n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scaling = scaling or ("energy" if scale_by_mask_ratio else "none")
        if self.scaling not in {"none", "energy", "unbiased"}:
            raise ValueError("scaling must be none, energy or unbiased")
        if self.scaling == "unbiased" and self.sparse_type != "randk":
            raise ValueError("unbiased scaling requires uniform randk sampling")
        self.scale_factor, self._indices = self.scale, None
        self._init_state(update_gap_scheduler, random_state)

    def should_update_projector(self, iteration):
        return self._update_due(iteration, self._indices is not None)

    @torch.no_grad()
    def project(self, full_grad, iteration):
        shape = self._check_input(full_grad)
        if self.should_update_projector(iteration):
            self._build_indices(full_grad)
        self._orig_shape = shape
        return full_grad.reshape(-1).index_select(0, self._indices)

    @torch.no_grad()
    def project_back(self, small_grad, output_buffer=None, alpha=1.0, accumulate=False):
        if self._indices is None:
            raise ValueError("project must be called before project_back")
        if small_grad.numel() != self._indices.numel():
            raise ValueError("compressed gradient has an incompatible shape")
        flat = small_grad.new_zeros(math.prod(self._orig_shape))
        flat.scatter_(0, self._indices, small_grad.reshape(-1) * self.scale_factor)
        return write_back(flat.reshape(self._orig_shape), output_buffer, alpha, accumulate)

    def _build_indices(self, x):
        flat = x.reshape(-1)
        k = kept_size(self.sparse_ratio, flat.numel())
        _, self._indices = self._create_sparse_mask(flat.abs(), self.sparse_type, k)
        p = k / flat.numel()
        divisor = math.sqrt(p) if self.scaling == "energy" else p if self.scaling == "unbiased" else 1
        self.scale_factor = self.scale / divisor
