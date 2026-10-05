"""Exact global residual-basis compensation for native GPT2 Conv1D modules."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.hypotheses.run_h03_synthetic import hadamard, orthogonal_qr_cleanup, quant4

CONVOLUTIONS = (("attn", "c_attn", "input"), ("attn", "c_proj", "output"),
                ("mlp", "c_fc", "input"), ("mlp", "c_proj", "output"))


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    model.config.attn_pdrop = model.config.resid_pdrop = model.config.embd_pdrop = 0.0
    model.config.use_cache = False
    return model.eval()


def untie_and_absorb(model, *, clone=True):
    """Absorb affine LN in row-oriented Conv1D and the final Linear head."""
    model = copy.deepcopy(model) if clone else model
    disable_dropout(model)
    if model.config.add_cross_attention:
        raise ValueError("cross-attention is outside the GPT2 protocol")
    head = model.lm_head
    width = model.config.n_embd
    untied = nn.Linear(width, head.out_features, bias=True, device=head.weight.device, dtype=head.weight.dtype)
    with torch.no_grad():
        final = model.transformer.ln_f
        untied.weight.copy_(head.weight * final.weight[None, :])
        untied.bias.copy_(head.weight @ final.bias)
        model.lm_head = untied
        model.config.tie_word_embeddings = False
        for block in model.transformer.h:
            for norm_name, parent_name, conv_name in (("ln_1", "attn", "c_attn"), ("ln_2", "mlp", "c_fc")):
                norm = getattr(block, norm_name)
                conv = getattr(getattr(block, parent_name), conv_name)
                original = conv.weight.clone()
                conv.weight.mul_(norm.weight[:, None])
                conv.bias.add_(norm.bias @ original)
                setattr(block, norm_name, nn.LayerNorm(width, eps=norm.eps, elementwise_affine=False,
                        device=conv.weight.device, dtype=conv.weight.dtype))
        model.transformer.ln_f = nn.LayerNorm(width, eps=final.eps, elementwise_affine=False,
                device=head.weight.device, dtype=head.weight.dtype)
    return model


def complement(width, reference):
    # Analytic Householder columns, H=I-2vv.T/(v.Tv), v=e0+1/sqrt(width).
    # This is the same mean-complement convention as QR of the positive unit
    # vector, without a full768x768 numerical QR to manufacture its complement.
    if width < 2:
        raise ValueError("residual width must be at least two")
    e = reference.new_empty(width, width - 1)
    e[0].fill_(-1 / math.sqrt(width))
    e[1:].fill_(-1 / (width + math.sqrt(width)))
    e[1:].diagonal().add_(1)
    return e


def lift(perpendicular, e, *, assembly_dtype=None):
    width = e.shape[0]
    # Analytic expansion of I+E@(P-I)@E.T for the canonical Householder E.
    # Dense E GEMMs accumulate error along its repeated constant entries on
    # CUDA at width768. Row/column reductions and rank-one corrections avoid
    # that accumulation, in the SAME FP32 precision. Global model compensation
    # still performs every actual dense Q weight/embedding multiplication.
    # Identity remains exact and the zero-Cayley derivative remains nonzero.
    output_dtype = perpendicular.dtype
    perpendicular = perpendicular.to(dtype=assembly_dtype or output_dtype)
    eye_perp = torch.eye(width - 1, device=e.device, dtype=perpendicular.dtype)
    difference = perpendicular - eye_perp
    rows, columns = difference.sum(1), difference.sum(0)
    total = (rows.sum() + columns.sum()) / 2
    a = 1 / math.sqrt(width)
    c = 1 / (width + math.sqrt(width))
    top = torch.cat(((1 + a*a*total).reshape(1), -a * (columns - c*total)))
    left = -a * (rows - c*total)
    bottom = perpendicular - c * (rows[:, None] + columns[None, :]) + c*c*total
    q = torch.cat((top[None, :], torch.cat((left[:, None], bottom), 1)), 0)
    # Closest Frobenius mean-preserving correction, implemented by sums and
    # rank-one updates, rather than materializing and multiplying dense P_mean.
    row_error, column_error = q.sum(1) - 1, q.sum(0) - 1
    total_error = (row_error.sum() + column_error.sum()) / 2
    q = q - row_error[:, None]/width - column_error[None, :]/width + total_error/(width*width)
    return q.to(dtype=output_dtype)


def choose_rotation(method, hidden, seed, e, *, assembly_dtype=None):
    size = e.shape[1]
    gen = torch.Generator(device=hidden.device).manual_seed(seed + 81000)
    if method in ("identity", "cayley"):
        return torch.eye(size + 1, device=hidden.device, dtype=hidden.dtype)
    elif method == "hadamard":
        sizes, remaining = [], size
        while remaining:
            block_size = 1 << (remaining.bit_length() - 1)
            sizes.append(block_size)
            remaining -= block_size
        p = torch.block_diag(*[hadamard(n, hidden) for n in sizes])
    elif method in ("haar", "procrustes"):
        p = orthogonal_qr_cleanup(torch.randn(size, size, generator=gen, device=hidden.device, dtype=hidden.dtype))
        if method == "procrustes":
            q0 = lift(p, e, assembly_dtype=assembly_dtype)
            target = quant4(hidden @ q0.T, channel_axis=1)
            x, y = hidden @ e, target @ e
            u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
            p = orthogonal_qr_cleanup(vh.T @ u.T)
    else:
        raise ValueError("unsupported rotation method")
    return lift(p, e, assembly_dtype=assembly_dtype)


@torch.no_grad()
def rotate_global_in_place(model, q):
    """Every residual branch, learned position, embedding and head shares Q."""
    model.transformer.wte.weight.copy_(model.transformer.wte.weight @ q.T)
    model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ q.T)
    for block in model.transformer.h:
        for parent, name, role in CONVOLUTIONS:
            conv = getattr(getattr(block, parent), name)
            conv.weight.copy_(q @ conv.weight if role == "input" else conv.weight @ q.T)
            if role == "output":
                conv.bias.copy_(conv.bias @ q.T)
    model.lm_head.weight.copy_(model.lm_head.weight @ q.T)
    return model


def logits(model, tokens):
    return model(input_ids=tokens, use_cache=False, return_dict=True).logits


@torch.no_grad()
def global_rotation_checks(base, q, tokens, *, original=None):
    rotated = rotate_global_in_place(copy.deepcopy(base), q).eval()
    exact = logits(base if original is None else original, tokens)
    actual = logits(rotated, tokens)
    width = q.shape[0]
    output_error = float((actual - exact).norm()) / max(float(exact.norm()), 1e-30)
    orth_error = float((q.T @ q - torch.eye(width, device=q.device, dtype=q.dtype)).norm()) / width
    mean_error = float((q @ q.new_ones(width) - 1).norm()) / math.sqrt(width)
    threshold = 1e-10 if q.dtype == torch.float64 else 1e-5
    if output_error > threshold or orth_error > 1e-5 or mean_error > threshold:
        raise ArithmeticError(f"global compensation failed: output={output_error}, orth={orth_error}, relative_Q1={mean_error}")
    return {"precompression_output_error": output_error, "orthogonality_error": orth_error,
            "mean_preservation_relative_error": mean_error, "output_threshold": threshold}


class QuantizedConv1D(nn.Module):
    """Native Conv1D orientation: output channels are weight dimension 1."""
    def __init__(self, original):
        super().__init__()
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.nf = original.nf

    def forward(self, x):
        weight = quant4(self.weight, ste=self.training, channel_axis=1)
        return (x @ weight + self.bias).reshape(x.shape[:-1] + (self.nf,))


class RotatedConv1D(QuantizedConv1D):
    def __init__(self, original, role):
        super().__init__(original)
        self.role = role
        self.q = None
        self.quantized = True

    def forward(self, x):
        if self.q is None:
            raise RuntimeError("rotation must be assigned before block forward")
        weight = self.q @ self.weight if self.role == "input" else self.weight @ self.q.T
        bias = self.bias if self.role == "input" else self.bias @ self.q.T
        if self.quantized:
            weight = quant4(weight, ste=self.training, channel_axis=1)
        return x @ weight + bias


class CalibrationBlock(nn.Module):
    def __init__(self, absorbed_block, q, *, cayley=False, assembly_dtype=None):
        super().__init__()
        self.block = copy.deepcopy(absorbed_block)
        self.register_buffer("fixed_q", q.detach().clone())
        self.register_buffer("e", complement(q.shape[0], q))
        self.cayley = cayley
        self.assembly_dtype = assembly_dtype
        self.quantized = True
        if cayley:
            self.skew_coordinates = nn.Parameter(q.new_zeros((q.shape[0] - 1) * (q.shape[0] - 2) // 2))
        for parent, name, role in CONVOLUTIONS:
            owner = getattr(self.block, parent)
            setattr(owner, name, RotatedConv1D(getattr(owner, name), role))

    def rotation(self):
        if not self.cayley:
            return self.fixed_q
        size = self.e.shape[1]
        rows, cols = torch.triu_indices(size, size, offset=1, device=self.e.device)
        a = self.e.new_zeros(size, size).index_put((rows, cols), self.skew_coordinates)
        a = a - a.T
        eye = torch.eye(size, device=a.device, dtype=a.dtype)
        return lift(torch.linalg.solve((eye + a).T, (eye - a).T).T, self.e, assembly_dtype=self.assembly_dtype)

    def forward(self, hidden):
        q = self.rotation()
        for parent, name, _ in CONVOLUTIONS:
            conv = getattr(getattr(self.block, parent), name)
            conv.q, conv.quantized = q, self.quantized
        return self.block(hidden @ q.T, use_cache=False)[0] @ q

    @torch.no_grad()
    def commit_unrotated_weights(self, target_block):
        for parent, name, _ in CONVOLUTIONS:
            source = getattr(getattr(self.block, parent), name)
            target = getattr(getattr(target_block, parent), name)
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)


def install_quantization(model, block_index=5):
    for parent, name, _ in CONVOLUTIONS:
        owner = getattr(model.transformer.h[block_index], parent)
        setattr(owner, name, QuantizedConv1D(getattr(owner, name)))
    return model


@torch.no_grad()
def cache_block_inputs(model, windows, device, *, block_index=5, batch_size=8, guard=None):
    captured = []
    def capture(_module, inputs):
        captured.append(inputs[0].detach().cpu())
    handle = model.transformer.h[block_index].register_forward_pre_hook(capture)
    try:
        for start in range(0, len(windows), batch_size):
            if guard:
                guard.check()
            logits(model, windows[start:start + batch_size].to(device))
    finally:
        handle.remove()
    hidden = torch.cat(captured)
    targets = []
    block = model.transformer.h[block_index]
    for start in range(0, len(hidden), batch_size):
        if guard:
            guard.check()
        targets.append(block(hidden[start:start + batch_size].to(device), use_cache=False)[0].cpu())
    return hidden, torch.cat(targets)


def small_admission():
    results = {}
    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(73)
        config = GPT2Config(n_embd=32, n_head=4, n_layer=3, n_positions=16, n_ctx=16,
                            vocab_size=37, resid_pdrop=0, attn_pdrop=0, embd_pdrop=0)
        original = disable_dropout(GPT2LMHeadModel(config).to(dtype=dtype))
        with torch.no_grad():
            for module in original.modules():
                if isinstance(module, nn.LayerNorm):
                    module.weight.copy_(torch.linspace(.7, 1.3, 32, dtype=dtype))
                    module.bias.copy_(torch.linspace(-.2, .2, 32, dtype=dtype))
            for block in original.transformer.h:
                block.attn.c_proj.bias.copy_(torch.linspace(-.1, .1, 32, dtype=dtype))
                block.mlp.c_proj.bias.copy_(torch.linspace(.2, -.2, 32, dtype=dtype))
        tokens = torch.randint(37, (2, 16), generator=torch.Generator().manual_seed(71))
        base = untie_and_absorb(original)
        assert base.lm_head.weight.data_ptr() != base.transformer.wte.weight.data_ptr()
        hidden, target = cache_block_inputs(base, tokens, "cpu", block_index=1, batch_size=2)
        e = complement(32, hidden)
        for method in ("identity", "hadamard", "haar", "procrustes", "cayley"):
            q = choose_rotation(method, hidden.reshape(-1, 32), 101, e)
            results[f"{dtype}_{method}"] = global_rotation_checks(base, q, tokens, original=original)
            calibrated = CalibrationBlock(base.transformer.h[1], q, cayley=method == "cayley")
            calibrated.quantized = False
            if calibrated.cayley:
                with torch.no_grad():
                    calibrated.skew_coordinates.copy_(.02 * torch.randn_like(calibrated.skew_coordinates))
            torch.testing.assert_close(calibrated(hidden), target, rtol=1e-10 if dtype == torch.float64 else 1e-5,
                                       atol=1e-10 if dtype == torch.float64 else 1e-5)
        changed = tokens.clone()
        changed[:, 7:] = (changed[:, 7:] + 1).remainder(37)
        torch.testing.assert_close(logits(original, tokens)[:, :7], logits(original, changed)[:, :7], rtol=0, atol=0)
    return results
