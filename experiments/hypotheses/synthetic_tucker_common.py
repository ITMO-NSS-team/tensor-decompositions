"""Native PyTorch reference operations shared by new H05/H06 toy runners."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.hypotheses.run_h02_synthetic import (
    ResourceGuard, minibatches, normalized_error, synchronize, tensor_hash,
)

SHAPE = (16, 8, 3, 3)
SIGNAL_RANKS = (4, 3, 2, 2)
SEEDS = (11, 22, 33, 44, 55)


def unfold(tensor, mode):
    return tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor, matrix, mode):
    if matrix.ndim != 2 or matrix.shape[1] != tensor.shape[mode]:
        raise ValueError("mode product shape mismatch")
    product = matrix @ unfold(tensor, mode)
    remaining = [size for k, size in enumerate(tensor.shape) if k != mode]
    return product.reshape(matrix.shape[0], *remaining).movedim(0, mode)


def reconstruct(core, factors):
    result = core
    for mode, factor in enumerate(factors):
        result = mode_dot(result, factor, mode)
    return result


def project(tensor, factors):
    result = tensor
    for mode, factor in enumerate(factors):
        result = mode_dot(result, factor.T, mode)
    return result


def left_svd(matrix,rank=None):
    """Complete left basis without a quadratic right basis for wide unfoldings."""
    complete=matrix.shape[0] > matrix.shape[1] and (rank is None or rank>min(matrix.shape))
    return torch.linalg.svd(matrix, full_matrices=complete)


def exact_hosvd(tensor, ranks):
    if len(ranks) != tensor.ndim or any(not 1 <= r <= n for n, r in zip(tensor.shape, ranks)):
        raise ValueError("invalid modal ranks")
    factors = []
    for mode, rank in enumerate(ranks):
        u, _, _ = left_svd(unfold(tensor, mode),rank)
        factors.append(u[:, :rank])
    return project(tensor, factors), factors


def relative_residual(tensor, reconstructed):
    norm = float(tensor.norm())
    absolute = float((tensor - reconstructed).norm())
    return (absolute / norm if norm else None), absolute


def signal_weight(seed, device="cpu", *, sigma=0.05, dtype=torch.float32, flat=False, scale=1.0):
    gen = torch.Generator(device=device).manual_seed(seed)
    if flat:
        signal = torch.randn(SHAPE, generator=gen, device=device, dtype=dtype)
    else:
        factors = [torch.linalg.qr(torch.randn(n, r, generator=gen, device=device, dtype=dtype)).Q
                   for n, r in zip(SHAPE, SIGNAL_RANKS)]
        core = torch.randn(SIGNAL_RANKS, generator=gen, device=device, dtype=dtype)
        signal = reconstruct(core, factors)
    signal = signal / signal.norm()
    noise = torch.randn(SHAPE, generator=gen, device=device, dtype=dtype)
    observed = signal + sigma * noise / noise.norm()
    return scale * observed, scale * signal, gen


def synthetic_cnn(seed, device="cpu", *, include_test=True, sigma=0.05, flat=False, scale=1.0, zero=False):
    weight, signal, gen = signal_weight(seed, device, sigma=sigma, flat=flat, scale=scale)
    if zero:
        weight, signal = torch.zeros_like(weight), torch.zeros_like(signal)
    teacher = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4)).to(device).eval()
    with torch.no_grad():
        for layer in (teacher[0], teacher[6]):
            fan_in = layer.weight[0].numel()
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen, device=device) / math.sqrt(fan_in))
            layer.bias.zero_()
        teacher[2].weight.copy_(weight)
        teacher[2].bias.zero_()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    u = torch.linalg.qr(torch.randn(256, 8, generator=gen, device=device)).Q
    scales = torch.tensor([3., 2., 1., .7, .5, .3, .2, .1], device=device)
    splits = {}
    for k, (name, count) in enumerate((("recovery", 4096), ("calibration", 512), ("tuning", 512), ("test", 1024)), 1):
        if name == "test" and not include_test:
            continue
        rng = torch.Generator(device=device).manual_seed(seed + 1000 * k)
        x = ((torch.randn(count, 8, generator=rng, device=device) * scales) @ u.T
             + .05 * torch.randn(count, 256, generator=rng, device=device)).reshape(count, 1, 16, 16)
        with torch.no_grad():
            y = torch.cat([teacher(x[start:start + 128]) for start in range(0, count, 128)])
        splits[name] = (x, y)
    return teacher, splits, signal


class TuckerWeightConv(nn.Module):
    """Learn factors/core; reconstruct a temporary weight on each forward pass.

    Spatial ranks may be smaller than three. Reconstruction cost and the
    temporary dense tensor are part of actual execution, not a free parameter.
    """
    def __init__(self, core, factors, bias):
        super().__init__()
        self.core = nn.Parameter(core.detach().clone())
        self.factors = nn.ParameterList([nn.Parameter(factor.detach().clone()) for factor in factors])
        self.bias = nn.Parameter(bias.detach().clone())
        self.ranks = tuple(core.shape)

    def weight(self):
        return reconstruct(self.core, self.factors)

    def forward(self, x):
        return F.conv2d(x, self.weight(), self.bias, padding=1)


def student_from(teacher, core, factors):
    student = copy.deepcopy(teacher)
    student[2] = TuckerWeightConv(core, factors, teacher[2].bias)
    return student.eval()


@torch.no_grad()
def cnn_metrics(student, teacher, pair):
    x, y = pair
    error = energy = 0.0
    count = agreement = 0
    for start in range(0, len(x), 128):
        prediction = student(x[start:start + 128])
        target = y[start:start + 128]
        if not torch.isfinite(prediction).all():
            raise ArithmeticError("nonfinite CNN prediction")
        error += float((prediction - target).square().sum())
        energy += float(target.square().sum())
        count += target.numel()
        agreement += int((prediction.argmax(1) == target.argmax(1)).sum())
    return {"mse": error / count, "normalized_mse": error / energy if energy else None,
            "zero_target_absolute_mse": error / count if not energy else None,
            "teacher_argmax_agreement": agreement / len(x), "n_examples": len(x)}


def recover_cnn(student, splits, seed, device, *, pilot=False, guard=None):
    histories = {}
    timing = {}
    # One optimizer spans both phases, identically for all methods.
    optimizer = torch.optim.AdamW(student[2].parameters(), lr=.001, weight_decay=0)
    for phase, steps, offset in (("calibration", 20 if pilot else 128, 50000),
                                 ("recovery", 0 if pilot else 512, 60000)):
        x, y = splits[phase]
        batches = minibatches(seed + offset, len(x), steps, device)
        synchronize(device)
        import time
        start = time.perf_counter()
        losses = []
        for ix in batches:
            if guard:
                guard.check()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(student(x[ix]), y[ix])
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite CNN training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student[2].parameters()):
                raise ArithmeticError("nonfinite CNN gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        synchronize(device)
        timing[phase + "_seconds"] = time.perf_counter() - start
        histories[phase] = {"steps": steps, "batches_sha256": tensor_hash(batches), "losses": losses}
    return histories, timing


def parameter_count(core, factors):
    return core.numel() + sum(factor.numel() for factor in factors)


def common_admission():
    """Independent einsum reference for axis order and dense autograd."""
    gen = torch.Generator().manual_seed(10705)
    core = torch.randn(4, 3, 2, 2, generator=gen, dtype=torch.float64)
    factors = [torch.randn(n, r, generator=gen, dtype=torch.float64) for n, r in zip(SHAPE, core.shape)]
    reference = torch.einsum("abcd,oa,ib,hc,wd->oihw", core, *factors)
    torch.testing.assert_close(reconstruct(core, factors), reference, atol=1e-12, rtol=1e-12)
    conv = TuckerWeightConv(core, factors, torch.zeros(16, dtype=torch.float64))
    xx = torch.randn(2, 8, 4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    independent_core = core.clone().requires_grad_()
    independent_factors = [factor.clone().requires_grad_() for factor in factors]
    yy = xx.detach().clone().requires_grad_()
    actual = conv(xx)
    ref_weight = torch.einsum("abcd,oa,ib,hc,wd->oihw", independent_core, *independent_factors)
    exact = F.conv2d(yy, ref_weight, padding=1)
    torch.testing.assert_close(actual, exact, atol=1e-10, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().mean(), (xx, conv.core, *conv.factors))
    exact_grad = torch.autograd.grad(exact.square().mean(), (yy, independent_core, *independent_factors))
    for a, b in zip(actual_grad, exact_grad):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-11)
    return {"native_mode_products_vs_einsum": "passed", "dense_tucker_output_and_gradients": "passed"}
