import math
import torch
import tensorly as tl
from tdecomp.tensor.tucker import HOOIDecomposition
from ._common import ProjectorState, rank_size, svd_name, validate_rank, write_back


def mode_product(tensor, matrix, mode):
    result = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    return result.movedim(0, mode)


class TensorGradLowRankProjector(ProjectorState):
    _config_fields = ("rank", "scale", "warm_restart", "n_iter_max", "svd_type")
    _data_fields = ("proj_tensor", "num_updates", "num_steps")

    def __init__(self, rank, update_gap_scheduler=None, verbose=False, scale=1.0,
                 warm_restart=False, n_iter_max=10, svd_type="truncated_svd",
                 tensor_decomposer_type=HOOIDecomposition, random_state=0):
        validate_rank(rank)
        if not isinstance(n_iter_max, int) or isinstance(n_iter_max, bool) or n_iter_max <= 0:
            raise ValueError("n_iter_max must be a positive integer")
        self.rank, self.verbose = rank, verbose
        self.scale = float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")
        self.warm_restart, self.n_iter_max, self.svd_type = warm_restart, n_iter_max, svd_name(svd_type)
        self.proj_tensor = None
        self.num_updates = self.num_steps = 0
        self.tensor_decomposer_type = tensor_decomposer_type
        self._init_state(update_gap_scheduler, random_state)

    def state_dict(self):
        if self.tensor_decomposer_type is not HOOIDecomposition:
            raise ValueError("custom tensor decomposer types cannot be saved in the portable state format; use HOOIDecomposition")
        return super().state_dict()

    def should_update_projector(self, iteration):
        return self._update_due(iteration, self.proj_tensor is not None)

    @torch.no_grad()
    def project(self, full_rank_grad, iter):
        shape = self._check_input(full_rank_grad)
        if isinstance(self.rank, (tuple, list)) and len(self.rank) != len(shape):
            raise ValueError("rank must have one entry per tensor mode")
        if self.should_update_projector(iter):
            self.proj_tensor = self.get_projection_tensor(full_rank_grad)
            self.num_updates += 1
        self._orig_shape, self.num_steps = shape, iter
        return self.transform(self.proj_tensor, full_rank_grad)

    @torch.no_grad()
    def project_back(self, low_rank_grad, output_buffer=None, alpha=1.0, accumulate=False):
        if self.proj_tensor is None:
            raise ValueError("project must be called before project_back")
        result = self.inverse_transform(self.proj_tensor, low_rank_grad) * self.scale
        return write_back(result, output_buffer, alpha, accumulate)

    def get_projection_tensor(self, weights):
        ranks = self.rank if isinstance(self.rank, (tuple, list)) else [self.rank] * weights.ndim
        ranks = [rank_size(rank, size) for rank, size in zip(ranks, weights.shape)]
        init = "svd"
        if self.warm_restart and self.proj_tensor is not None:
            init = (self.transform(self.proj_tensor, weights), self.proj_tensor)
        # Temporary backend context restores the caller's TensorLy backend.
        with tl.backend_context("pytorch"):
            decomposer = self.tensor_decomposer_type(rank=ranks, init=init,
                n_iter_max=self.n_iter_max, svd_type="truncated_svd" if self.svd_type == "full_svd" else self.svd_type)
            _, factors = decomposer.decompose(weights, random_state=self._seed())
        return [factor.to(device=weights.device, dtype=weights.dtype) for factor in factors]

    @torch.no_grad()
    def transform(self, proj_tensor, full_rank_grad):
        result = full_rank_grad
        for mode, factor in enumerate(proj_tensor):
            result = mode_product(result, factor.mH, mode)
        return result

    @torch.no_grad()
    def inverse_transform(self, proj_tensor, x, output_buffer=None, alpha=1.0, accumulate=False):
        result = x
        for mode, factor in enumerate(proj_tensor):
            result = mode_product(result, factor, mode)
        return write_back(result, output_buffer, alpha, accumulate)
