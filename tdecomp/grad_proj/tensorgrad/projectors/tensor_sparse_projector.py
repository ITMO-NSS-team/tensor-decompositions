import math
import torch
from .abstract_sparce_projector import AbstractSparceProjector
from ._common import kept_size, sparse_name, validate_ratio, write_back


def mode_unfolding_norms(tensor, mode):
    return torch.linalg.vector_norm(tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1), dim=1)


class TensorGradSparseProjector(AbstractSparceProjector):
    """Gather a Cartesian product of mode indices; scalar ratio is distributed over nonunit axes."""
    _config_fields = ("sparse_ratio", "sparse_type", "scale", "scale_by_mask_ratio", "scaling")
    _data_fields = ("masks", "indices", "scale_factor")

    def __init__(self, sparse_ratio=0.25, sparse_type="topk", verbose=False,
                 update_gap_scheduler=None, scale=1.0, warm_restart=False,
                 n_iter_max=10, scale_by_mask_ratio=False, scaling=None, random_state=0):
        validate_ratio(sparse_ratio)
        self.sparse_ratio, self.sparse_type = sparse_ratio, sparse_name(sparse_type)
        self.scale = float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")
        self.verbose, self.warm_restart, self.n_iter_max = verbose, warm_restart, n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scaling = scaling or ("energy" if scale_by_mask_ratio else "none")
        if self.scaling not in {"none", "energy", "unbiased"}:
            raise ValueError("scaling must be none, energy or unbiased")
        if self.scaling == "unbiased" and self.sparse_type != "randk":
            raise ValueError("unbiased scaling requires uniform randk sampling")
        self.masks = self.indices = None
        self.scale_factor = self.scale
        self._init_state(update_gap_scheduler, random_state)

    def should_update_projector(self, iteration):
        return self._update_due(iteration, self.masks is not None)

    @torch.no_grad()
    def project(self, full_rank_grad, iteration):
        shape = self._check_input(full_rank_grad)
        if isinstance(self.sparse_ratio, (tuple, list)) and len(self.sparse_ratio) != len(shape):
            raise ValueError("sparse_ratio must have one entry per mode")
        if self.should_update_projector(iteration):
            self._build_masks(full_rank_grad)
        self._orig_shape = shape
        return self._transform(full_rank_grad)

    def _build_masks(self, tensor):
        if isinstance(self.sparse_ratio, (tuple, list)):
            ratios = self.sparse_ratio
        else:
            count = sum(size != 1 for size in tensor.shape)
            ratio = self.sparse_ratio ** (1 / count) if count else 1.0
            ratios = [ratio if size != 1 else 1.0 for size in tensor.shape]
        masks, indices = [], []
        for mode, (size, ratio) in enumerate(zip(tensor.shape, ratios)):
            scores = None if self.sparse_type == "randk" else mode_unfolding_norms(tensor, mode)
            mask, index = self._create_sparse_mask(scores, self.sparse_type, kept_size(ratio, size),
                                                  {"device": tensor.device}, dim_size=size)
            masks.append(mask)
            indices.append(index)
        self.masks, self.indices = masks, indices
        p = math.prod(index.numel() for index in indices) / tensor.numel()
        divisor = math.sqrt(p) if self.scaling == "energy" else p if self.scaling == "unbiased" else 1
        self.scale_factor = self.scale / divisor
        return masks

    def _transform(self, tensor):
        result = tensor
        for mode, index in enumerate(self.indices):
            result = result.index_select(mode, index)
        return result

    @torch.no_grad()
    def project_back(self, small_grad, output_buffer=None, alpha=1.0, accumulate=False):
        if self.indices is None:
            raise ValueError("project must be called before project_back")
        if tuple(small_grad.shape) != tuple(x.numel() for x in self.indices):
            raise ValueError("compressed gradient has an incompatible shape")
        result = small_grad * self.scale_factor
        for mode in reversed(range(len(self.indices))):
            shape = list(result.shape)
            shape[mode] = self._orig_shape[mode]
            bigger = result.new_zeros(shape)
            bigger.index_copy_(mode, self.indices[mode], result)
            result = bigger
        return write_back(result, output_buffer, alpha, accumulate)

    _inverse_transform = project_back
