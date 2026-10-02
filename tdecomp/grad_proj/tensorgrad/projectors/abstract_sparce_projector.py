import torch
from ._common import ProjectorState, sparse_name


class AbstractSparceProjector(ProjectorState):
    def _create_sparse_mask(self, scores, sparse_type, k, context=None, dim_size=None):
        """Select k unique positions; probability sampling falls back on uniform zero scores."""
        sparse_type = sparse_name(sparse_type)
        dim_size = scores.numel() if dim_size is None and scores is not None else dim_size
        if dim_size is None or not 1 <= k <= dim_size:
            raise ValueError("selection requires a dimension length and 1 <= k <= length")
        device = scores.device if scores is not None else (context or {}).get("device", "cpu")
        if sparse_type == "topk":
            if scores is None:
                raise ValueError("topk requires scores")
            indices = torch.argsort(scores, descending=True, stable=True)[:k].cpu()
        elif sparse_type == "randk":
            indices = torch.randperm(dim_size, generator=self.generator)[:k]
        else:
            if scores is None or not torch.isfinite(scores).all() or (scores < 0).any():
                raise ValueError("probability requires finite nonnegative scores")
            weights = scores.detach().to(device="cpu", dtype=torch.float64)
            # All coordinates retain positive weight, even for zero-valued slices.
            weights = weights + torch.finfo(weights.dtype).eps * max(1.0, float(weights.max()))
            indices = torch.multinomial(weights, k, replacement=False, generator=self.generator)
        indices = indices.sort().values.to(device=device)
        mask = torch.zeros(dim_size, device=device, dtype=torch.bool)
        mask[indices] = True
        return mask, indices
