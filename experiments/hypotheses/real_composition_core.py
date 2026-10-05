"""H02 two-convolution factorization with the original full-width nonlinear graph."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F


class ComposedBlock(nn.Module):
    def __init__(self, original, rank, *, shared_initialization=False):
        super().__init__()
        if original.downsample is not None or original.conv1.stride != (1, 1):
            raise ValueError('H02 requires an identity residual and stride one')
        w1, w2 = original.conv1.weight.detach(), original.conv2.weight.detach()
        n = w1.shape[0]
        m1 = w1.reshape(n, -1)
        m2 = w2.permute(0, 2, 3, 1).reshape(-1, n)
        if shared_initialization:
            _, q = torch.linalg.eigh(m1 @ m1.T + m2.T @ m2)
            q1 = q[:, -rank:].flip(1)
            q2 = q1.clone()
        else:
            q1 = torch.linalg.svd(m1, full_matrices=False).U[:, :rank]
            q2 = torch.linalg.svd(m2, full_matrices=False).Vh[:rank].T
        self.a1 = nn.Parameter((q1.T @ m1).reshape(rank, n, 3, 3).contiguous())
        self.q1 = nn.Parameter(q1.clone().contiguous())
        self.q2 = nn.Parameter(q2.clone().contiguous())
        self.a2 = nn.Parameter((m2 @ q2).reshape(n, 3, 3, rank).permute(0, 3, 1, 2).contiguous())
        self.bn1, self.bn2 = copy.deepcopy(original.bn1), copy.deepcopy(original.bn2)
        for p in self.bn1.parameters(): p.requires_grad_(False)
        for p in self.bn2.parameters(): p.requires_grad_(False)

    def factors(self):
        return (self.a1, self.q1, self.q2, self.a2)

    def first(self, x):
        return F.conv2d(F.conv2d(x, self.a1, padding=1), self.q1[:, :, None, None])

    def second(self, x):
        return F.conv2d(F.conv2d(x, self.q2.T[:, :, None, None]), self.a2, padding=1)

    def forward(self, x):
        hidden = F.relu(self.bn1(self.first(x)), inplace=False)
        return F.relu(self.bn2(self.second(hidden)) + x, inplace=False)


def admission():
    """Full-rank output/gradient identity and independent dense factor reference."""
    from torchvision.models.resnet import BasicBlock
    records = []
    torch.manual_seed(20261005)
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-5)):
        original = BasicBlock(8, 8).to(dtype).eval()
        student = ComposedBlock(original, 8).eval()
        x = torch.randn(2, 8, 2, 2, dtype=dtype, requires_grad=True)
        y0, y = original(x), student(x)
        relative = float(torch.linalg.vector_norm(y-y0) / torch.linalg.vector_norm(y0))
        assert relative <= tolerance
        g0, = torch.autograd.grad(y0.square().sum(), x, retain_graph=True)
        g, = torch.autograd.grad(y.square().sum(), x)
        gradient_error = float(torch.linalg.vector_norm(g-g0) / torch.linalg.vector_norm(g0))
        assert gradient_error <= tolerance
        compressed = ComposedBlock(original, 3).eval()
        w1 = torch.einsum('or,rihw->oihw', compressed.q1, compressed.a1)
        w2 = torch.einsum('orhw,ir->oihw', compressed.a2, compressed.q2)
        reference = F.relu(compressed.bn2(F.conv2d(
            F.relu(compressed.bn1(F.conv2d(x, w1, padding=1))), w2, padding=1)) + x)
        actual = compressed(x)
        assert torch.allclose(actual, reference, rtol=tolerance, atol=tolerance)
        for parameter in (x, *compressed.factors()):
            ga, = torch.autograd.grad(actual.square().sum(), parameter, retain_graph=True)
            gr, = torch.autograd.grad(reference.square().sum(), parameter, retain_graph=True)
            assert torch.allclose(ga, gr, rtol=tolerance*10, atol=tolerance*10)
        assert sum(p.numel() for p in compressed.factors()) == 2*8*9*3 + 2*8*3
        shared = ComposedBlock(original, 3, shared_initialization=True)
        assert shared.q1.data_ptr() != shared.q2.data_ptr()
        records.append({'dtype': str(dtype), 'output_error': relative,
                        'input_gradient_error': gradient_error, 'factor_gradient_reference': 'passed'})
    return records
