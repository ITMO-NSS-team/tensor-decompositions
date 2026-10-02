import torch
from .tensor_sparse_projector import TensorGradSparseProjector


class GaLoreSparseProjector(TensorGradSparseProjector):
    """Matrix row/column sampler, retaining the public GaLore sparse orientations."""
    _config_fields = TensorGradSparseProjector._config_fields + ("proj_type",)

    def __init__(self, sparse_ratio=0.25, sparse_type="topk", verbose=False,
                 update_gap_scheduler=None, scale=1.0, proj_type="std",
                 activation_checkpoint=False, random_state=0):
        if proj_type not in {"std", "reverse_std", "left", "right"}:
            raise ValueError("proj_type must be std, reverse_std, left or right")
        if isinstance(sparse_ratio, (tuple, list)):
            raise ValueError("GaLoreSparseProjector sparse_ratio must be scalar")
        super().__init__(sparse_ratio, sparse_type, verbose, update_gap_scheduler,
                         scale, random_state=random_state)
        self.proj_type, self.activation_checkpoint = proj_type, activation_checkpoint

    @torch.no_grad()
    def project(self, full_rank_grad, iteration):
        if full_rank_grad.ndim != 2:
            raise ValueError("GaLoreSparseProjector requires a matrix")
        original_ratio = self.sparse_ratio
        do_rows = self.proj_type == "left" or (
            self.proj_type == "std" and full_rank_grad.shape[0] < full_rank_grad.shape[1]) or (
            self.proj_type == "reverse_std" and full_rank_grad.shape[0] >= full_rank_grad.shape[1])
        self.sparse_ratio = [original_ratio, 1.0] if do_rows else [1.0, original_ratio]
        try:
            result = super().project(full_rank_grad, iteration)
        finally:
            self.sparse_ratio = original_ratio
        self._mask = self.masks[0 if do_rows else 1]
        return result
