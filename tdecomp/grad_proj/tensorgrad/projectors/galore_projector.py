import math
import torch
from ._common import ProjectorState, rank_size, svd_name, validate_rank, write_back


class GaLoreProjector(ProjectorState):
    """Matrix SVD projector, optionally flattening higher modes after the first."""
    _config_fields = ("rank", "svd_type", "scale", "galore_2d_proj_type", "activation_checkpoint", "support_complex")
    _data_fields = ("ortho_matrix",)

    def __init__(self, rank, verbose=False, svd_type=None, update_gap_scheduler=None,
                 scale=1.0, galore_2d_proj_type="left", activation_checkpoint=False,
                 support_complex=False, random_state=0):
        validate_rank(rank)
        if isinstance(rank, (tuple, list)):
            raise ValueError("GaLore rank must be scalar")
        if galore_2d_proj_type not in {"left", "right", "full"}:
            raise ValueError("galore_2d_proj_type must be left, right or full")
        self.rank, self.verbose, self.svd_type = rank, verbose, svd_name(svd_type)
        self.scale = float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")
        self.galore_2d_proj_type = galore_2d_proj_type
        self.activation_checkpoint = self.activation_checkpointing = activation_checkpoint
        self.support_complex, self.ortho_matrix = support_complex, None
        self._init_state(update_gap_scheduler, random_state)

    def _svd(self, matrix, rank):
        if callable(self.svd_type):
            import tensorly as tl
            with tl.backend_context("pytorch"):
                return self.svd_type(matrix)
        if self.svd_type != "randomized_svd" or rank >= min(matrix.shape):
            return torch.linalg.svd(matrix, full_matrices=False)
        # Local Gaussian range finder, then SVD in that orthonormal range.
        size = min(min(matrix.shape), rank + 5)
        omega = torch.randn(matrix.shape[1], size, dtype=matrix.dtype, generator=self.generator)
        omega = omega.to(matrix.device)
        q = torch.linalg.qr(matrix @ omega, mode="reduced").Q
        for _ in range(2):
            z = torch.linalg.qr(matrix.mH @ q, mode="reduced").Q
            q = torch.linalg.qr(matrix @ z, mode="reduced").Q
        u, s, vh = torch.linalg.svd(q.mH @ matrix, full_matrices=False)
        return q @ u, s, vh

    @torch.no_grad()
    def project(self, full_rank_grad, iter):
        with torch.profiler.record_function("tdecomp.galore.project"):
            shape = self._check_input(full_rank_grad)
            if full_rank_grad.is_complex() and not self.support_complex:
                raise ValueError("complex projection requires support_complex=True")
            matrix = full_rank_grad.reshape(shape[0], -1)
            if self._update_due(iter, self.ortho_matrix is not None):
                self.ortho_matrix = self.get_orthogonal_matrix(matrix, self.rank, self.galore_2d_proj_type)
            self._orig_shape = shape
            if self.galore_2d_proj_type == "left":
                return optional_checkpoint_matmul(self.ortho_matrix.mH, matrix, self.activation_checkpoint)
            if self.galore_2d_proj_type == "right":
                return optional_checkpoint_matmul(matrix, self.ortho_matrix.mH, self.activation_checkpoint)
            intermediate = optional_checkpoint_matmul(self.ortho_matrix[0].mH, matrix, self.activation_checkpoint)
            return optional_checkpoint_matmul(intermediate, self.ortho_matrix[1].mH, self.activation_checkpoint)

    @torch.no_grad()
    def project_back(self, low_rank_grad, output_buffer=None, alpha=1.0, accumulate=False):
        with torch.profiler.record_function("tdecomp.galore.project_back"):
            if self.ortho_matrix is None:
                raise ValueError("project must be called before project_back")
            if self.galore_2d_proj_type == "left":
                result = optional_checkpoint_matmul(self.ortho_matrix, low_rank_grad, self.activation_checkpoint)
            elif self.galore_2d_proj_type == "right":
                result = optional_checkpoint_matmul(low_rank_grad, self.ortho_matrix, self.activation_checkpoint)
            else:
                intermediate = optional_checkpoint_matmul(self.ortho_matrix[0], low_rank_grad, self.activation_checkpoint)
                result = optional_checkpoint_matmul(intermediate, self.ortho_matrix[1], self.activation_checkpoint)
            result = (result * self.scale).reshape(self._orig_shape)
            return write_back(result, output_buffer, alpha, accumulate)

    @torch.no_grad()
    def get_orthogonal_matrix(self, tensor, rank, galore2dProjectionSide):
        count = rank_size(rank, min(tensor.shape))
        u, _, vh = self._svd(tensor, count)
        if galore2dProjectionSide == "left":
            return u[:, :count].clone()
        if galore2dProjectionSide == "right":
            return vh[:count, :].clone()
        if galore2dProjectionSide == "full":
            return [u[:, :count].clone(), vh[:count, :].clone()]
        raise ValueError("projection side must be left, right or full")


def optional_checkpoint_matmul(a, b, activation_checkpoint=True):
    if activation_checkpoint:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(torch.matmul, a, b, use_reentrant=False)
    return torch.matmul(a, b)
