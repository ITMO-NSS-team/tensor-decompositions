"""H03 mean-preserving rotations, exact Transformer compensation and int4 QAT.

Attention head coordinates and FFN GELU coordinates stay unchanged. LayerNorm
affine terms are absorbed into its input projections before any rotation.
Quantization executes dequantized weights; no native int4 speedup is claimed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from torch import nn
from torch.nn import functional as F
from experiments.hypotheses.run_h02_synthetic import (
    BASE_SHA, ResourceGuard, file_hash, git, minibatches, synchronize, tensor_hash, write_csv, write_json,
)

WIDTH = 64
VOCABULARY = 32
METHODS = ("identity", "hadamard", "haar", "procrustes", "cayley")
SEEDS = (11, 22, 33, 44, 55)
QUANTIZED = ("query", "key", "value", "output", "ff1", "ff2")


class ToyTransformer(nn.Module):
    def __init__(self, seed=0, device="cpu", dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(32, 64)
        self.positions = nn.Parameter(torch.empty(32, 64))
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        for name in ("query", "key", "value", "output"):
            setattr(self, name, nn.Linear(64, 64, bias=False))
        self.ff1 = nn.Linear(64, 128, bias=False)
        self.ff2 = nn.Linear(128, 64, bias=False)
        self.head = nn.Linear(64, 32, bias=False)
        self.to(device=device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("norm1.weight") or name.endswith("norm2.weight"):
                    parameter.fill_(1)
                elif name.endswith("norm1.bias") or name.endswith("norm2.bias"):
                    parameter.zero_()
                else:
                    parameter.copy_(.02 * torch.randn(parameter.shape, generator=gen, device=device, dtype=dtype))
            self.head.weight.copy_(self.embedding.weight)

    def attend(self, x, weights=None, biases=None):
        weights = weights or {name: getattr(self, name).weight for name in ("query", "key", "value", "output")}
        biases = biases or {name: getattr(self, name).bias for name in ("query", "key", "value", "output")}
        b, length, _ = x.shape
        q, k, v = [F.linear(x, weights[name], biases[name]).reshape(b, length, 4, 16).transpose(1, 2)
                   for name in ("query", "key", "value")]
        scores = q @ k.transpose(-1, -2) / 4
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
        values = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, length, 64)
        return F.linear(values, weights["output"], biases["output"])

    def block(self, hidden):
        hidden = hidden + self.attend(self.norm1(hidden))
        return hidden + self.ff2(F.gelu(self.ff1(self.norm2(hidden))))

    def input_hidden(self, tokens):
        return self.embedding(tokens) + self.positions[:tokens.shape[1]]

    def forward(self, tokens):
        return self.head(self.block(self.input_hidden(tokens)))


def absorb_affine(teacher):
    model = copy.deepcopy(teacher)
    with torch.no_grad():
        for norm_name, names in (("norm1", ("query", "key", "value")), ("norm2", ("ff1",))):
            norm = getattr(teacher, norm_name)
            for name in names:
                original = getattr(teacher, name)
                linear = nn.Linear(original.in_features, original.out_features, bias=True,
                                   device=original.weight.device, dtype=original.weight.dtype)
                linear.weight.copy_(original.weight * norm.weight[None, :])
                linear.bias.copy_(original.weight @ norm.bias + (original.bias if original.bias is not None else 0))
                setattr(model, name, linear)
            setattr(model, norm_name, nn.LayerNorm(64, eps=norm.eps, elementwise_affine=False,
                                                 device=norm.weight.device, dtype=norm.weight.dtype))
    return model


def mean_complement(device="cpu", dtype=torch.float32):
    unit = torch.ones(64, 1, device=device, dtype=dtype) / 8
    return torch.linalg.qr(unit, mode="complete").Q[:, 1:].contiguous()


def lift_rotation(perpendicular, complement):
    return complement @ perpendicular @ complement.T + complement.new_ones(64, 64) / 64


def hadamard(size, reference):
    matrix = reference.new_ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    if matrix.shape[0] != size:
        raise ValueError("Hadamard size must be a power of two")
    return matrix / math.sqrt(size)


def orthogonal_qr_cleanup(matrix):
    """Remove numerical SVD orthogonality loss without changing QR signs.

    For an already orthogonal matrix, adjusted QR returns the same matrix.
    The cleanup runs in the input precision and is included in setup cost.
    """
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(r.diagonal() < 0, -torch.ones_like(r.diagonal()), torch.ones_like(r.diagonal()))
    return q * signs


def quant4(weight, *, ste=False, channel_axis=0):
    reduction = tuple(axis for axis in range(weight.ndim) if axis != channel_axis)
    scale = weight.detach().abs().amax(dim=reduction, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = weight / safe
    rounded = normalized + (normalized.round() - normalized).detach() if ste else normalized.round()
    result = rounded.clamp(-7, 7) * safe
    return torch.where(scale > 0, result, torch.zeros_like(result))


def fixed_rotation(method, hidden, seed, complement):
    gen = torch.Generator(device=hidden.device).manual_seed(seed + 81000)
    size = 63
    if method in ("identity", "cayley"):
        perpendicular = torch.eye(size, device=hidden.device, dtype=hidden.dtype)
    elif method == "hadamard":
        perpendicular = torch.block_diag(*[hadamard(size, hidden) for size in (32, 16, 8, 4, 2, 1)])
    elif method == "haar":
        perpendicular = orthogonal_qr_cleanup(torch.randn(size, size, generator=gen, device=hidden.device, dtype=hidden.dtype))
    elif method == "procrustes":
        initial = orthogonal_qr_cleanup(torch.randn(size, size, generator=gen, device=hidden.device, dtype=hidden.dtype))
        q0 = lift_rotation(initial, complement)
        target = quant4(hidden @ q0.T, channel_axis=1)
        x, y = hidden @ complement, target @ complement
        u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
        perpendicular = orthogonal_qr_cleanup(vh.T @ u.T)
    else:
        raise ValueError("unsupported rotation method")
    return lift_rotation(perpendicular, complement)


class RotatedQuantized(nn.Module):
    def __init__(self, absorbed, q, *, cayley=False, quantized=True):
        super().__init__()
        self.base = copy.deepcopy(absorbed)
        self.register_buffer("fixed_q", q.clone())
        self.register_buffer("complement", mean_complement(q.device, q.dtype))
        self.quantized = quantized
        self.cayley = cayley
        if cayley:
            self.skew_coordinates = nn.Parameter(q.new_zeros(63 * 62 // 2))

    def rotation(self):
        if not self.cayley:
            return self.fixed_q
        rows, cols = torch.triu_indices(63, 63, offset=1, device=self.fixed_q.device)
        a = self.fixed_q.new_zeros(63, 63).index_put((rows, cols), self.skew_coordinates)
        a = a - a.T
        eye = torch.eye(63, device=a.device, dtype=a.dtype)
        perpendicular = torch.linalg.solve((eye + a).T, (eye - a).T).T
        return lift_rotation(perpendicular, self.complement)

    def rotated_weights(self, q=None, *, ste=False):
        q = self.rotation() if q is None else q
        result = {}
        for name in QUANTIZED:
            original = getattr(self.base, name).weight
            rotated = q @ original if name in ("output", "ff2") else original @ q.T
            result[name] = quant4(rotated, ste=ste) if self.quantized else rotated
        return result

    def block(self, hidden_rotated, *, ste=False, q=None):
        q = self.rotation() if q is None else q
        weights = self.rotated_weights(q, ste=ste)
        biases = {name: getattr(self.base, name).bias for name in ("query", "key", "value", "output")}
        hidden_rotated = hidden_rotated + self.base.attend(self.base.norm1(hidden_rotated), weights, biases)
        ff = F.linear(self.base.norm2(hidden_rotated), weights["ff1"], self.base.ff1.bias)
        return hidden_rotated + F.linear(F.gelu(ff), weights["ff2"], self.base.ff2.bias)

    def forward(self, tokens, *, ste=False):
        q = self.rotation()
        hidden = (self.base.embedding(tokens) + self.base.positions[:tokens.shape[1]]) @ q.T
        output = self.block(hidden, ste=ste, q=q)
        return F.linear(output, self.base.head.weight @ q.T, self.base.head.bias)


def sequence_data(seed, device="cpu", *, include_test=True):
    splits = {}
    for k, (name, count) in enumerate((("recovery", 2560), ("calibration", 512), ("tuning", 512), ("test", 512)), 1):
        if name == "test" and not include_test:
            continue
        gen = torch.Generator(device=device).manual_seed(seed + 1000 * k)
        tokens = torch.empty(count, 33, device=device, dtype=torch.long)
        tokens[:, 0] = torch.randint(32, (count,), generator=gen, device=device)
        for step in range(1, 33):
            continuation = torch.rand(count, generator=gen, device=device) < .9
            random = torch.randint(32, (count,), generator=gen, device=device)
            tokens[:, step] = torch.where(continuation, (tokens[:, step - 1] + 1).remainder(32), random)
        splits[name] = tokens[:, :-1], tokens[:, 1:]
    return splits


@torch.no_grad()
def quality(model, pair, teacher=None):
    x, y = pair
    total = error = energy = 0.0
    for start in range(0, len(x), 32):
        predicted = model(x[start:start + 32])
        target = y[start:start + 32]
        if not torch.isfinite(predicted).all():
            raise ArithmeticError("nonfinite Transformer logits")
        total += float(F.cross_entropy(predicted.reshape(-1, 32), target.reshape(-1), reduction="sum"))
        if teacher is not None:
            exact = teacher(x[start:start + 32])
            error += float((predicted - exact).square().sum())
            energy += float(exact.square().sum())
    cross_entropy = total / y.numel()
    return {"cross_entropy": cross_entropy, "perplexity": math.exp(cross_entropy),
            "logits_normalized_mse": error / energy if energy else None, "sequences": len(x), "tokens": y.numel()}


def rotation_checks(student, teacher, tokens):
    with torch.no_grad():
        q = student.rotation()
        eye = torch.eye(64, device=q.device, dtype=q.dtype)
        orthogonal = float((q.T @ q - eye).norm()) / 64
        ones_error = float((q @ torch.ones(64, device=q.device, dtype=q.dtype) - 1).norm())
        quantized = student.quantized
        student.quantized = False
        try:
            actual, exact = student(tokens), teacher(tokens)
        finally:
            student.quantized = quantized
        error = float((actual - exact).norm()) / max(float(exact.norm()), 1e-30)
        threshold = 1e-10 if q.dtype == torch.float64 else 1e-5
        if error > threshold or orthogonal > 1e-5 or ones_error > 1e-5:
            raise ArithmeticError(f"exact rotation admission failed: output={error}, orthogonality={orthogonal}, ones={ones_error}")
        return {"precompression_output_error": error, "orthogonality_error": orthogonal,
                "mean_preservation_error": ones_error, "threshold": threshold}


def admission():
    results = {}
    for dtype in (torch.float64, torch.float32):
        teacher = ToyTransformer(37, dtype=dtype).eval()
        assert sum(parameter.numel() for parameter in teacher.parameters()) == 39168
        # Nontrivial learned affine parameters test actual absorption.
        with torch.no_grad():
            teacher.norm1.weight.copy_(torch.linspace(.7, 1.3, 64, dtype=dtype))
            teacher.norm1.bias.copy_(torch.linspace(-.2, .2, 64, dtype=dtype))
            teacher.norm2.weight.copy_(torch.linspace(1.2, .8, 64, dtype=dtype))
            teacher.norm2.bias.copy_(torch.linspace(.1, -.1, 64, dtype=dtype))
        tokens = torch.randint(32, (2, 32), generator=torch.Generator().manual_seed(93))
        absorbed = absorb_affine(teacher)
        hidden = teacher.input_hidden(tokens).reshape(-1, 64).detach()
        complement = mean_complement(dtype=dtype)
        for method in METHODS:
            q = fixed_rotation(method, hidden, 37, complement)
            student = RotatedQuantized(absorbed, q, cayley=method == "cayley", quantized=False)
            results[f"{dtype}_{method}"] = rotation_checks(student, teacher, tokens)
        original = teacher(tokens)
        changed = tokens.clone()
        changed[:, 10:] = changed[:, 10:].add(7).remainder(32)
        torch.testing.assert_close(teacher(changed)[:, :10], original[:, :10], rtol=0, atol=0)
    zero = torch.zeros(3, 4, dtype=torch.float64)
    assert torch.equal(quant4(zero), zero)
    values = torch.tensor([[0., 1., -1., .5], [0., 0., 0., 0.]], dtype=torch.float64)
    quantized = quant4(values)
    torch.testing.assert_close(quantized[0] * 7, (values[0] * 7).round())
    arbitrary = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(17), dtype=torch.float64)).Q
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    ln_error = float((F.layer_norm(x @ arbitrary.T, (64,)) - F.layer_norm(x, (64,)) @ arbitrary.T).norm())
    assert ln_error > .1
    rms = lambda v: v / (v.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    torch.testing.assert_close(rms(x @ arbitrary.T), rms(x) @ arbitrary.T, rtol=1e-12, atol=1e-12)
    def rope(position):
        matrices = []
        for j in range(8):
            angle = position * 10000**(-2 * j / 16)
            matrices.append(torch.tensor([[math.cos(angle), -math.sin(angle)],
                                          [math.sin(angle), math.cos(angle)]], dtype=torch.float64))
        return torch.block_diag(*matrices)
    allowed_head = torch.block_diag(*[torch.tensor([[math.cos(.3 + j), -math.sin(.3 + j)],
                      [math.sin(.3 + j), math.cos(.3 + j)]], dtype=torch.float64) for j in range(8)])
    forbidden_head = arbitrary[:16, :16]
    forbidden_head = torch.linalg.qr(forbidden_head).Q
    for position in (1, 7):
        matrix = rope(position)
        torch.testing.assert_close(allowed_head @ matrix, matrix @ allowed_head, atol=1e-12, rtol=1e-12)
        assert float((forbidden_head @ matrix - matrix @ forbidden_head).norm()) > .1
    results.update(causal_mask_future_independence="passed", zero_channel_quantization="passed",
                   arbitrary_LN_rotation_counterexample_norm=ln_error, pure_RMSNorm_orthogonal_control="passed",
                   rope_head_commutator_positive_and_negative_controls="passed")
    return results


def pretrain(teacher, pair, seed, device, steps=500, guard=None):
    x, y = pair
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=.001, weight_decay=.01)
    indices = minibatches(seed + 40000, len(x), steps, device, batch_size=16)
    losses = []
    for ix in indices:
        if guard:
            guard.check()
        optimizer.zero_grad(set_to_none=True)
        logits = teacher(x[ix])
        loss = F.cross_entropy(logits.reshape(-1, 32), y[ix].reshape(-1))
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite teacher loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"steps": steps, "batches_sha256": tensor_hash(indices), "losses": losses}


def calibrate(student, teacher, pair, seed, device, steps=50, guard=None):
    # Same six weight matrices and absorbed input biases for every method.
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for name in QUANTIZED:
        for parameter in getattr(student.base, name).parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if student.cayley:
        student.skew_coordinates.requires_grad_(True)
        parameters.append(student.skew_coordinates)
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0)
    x, _ = pair
    indices = minibatches(seed + 50000, len(x), steps, device, batch_size=16)
    losses = []
    for ix in indices:
        if guard:
            guard.check()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            hidden = teacher.input_hidden(x[ix])
            exact = teacher.block(hidden)
        q = student.rotation()
        predicted = student.block(hidden @ q.T, q=q, ste=True) @ q
        loss = F.mse_loss(predicted, exact)
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite calibration loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"steps": steps, "batches_sha256": tensor_hash(indices), "losses": losses,
            "extra_cayley_coordinates": student.skew_coordinates.numel() if student.cayley else 0}


def recover(student, teacher, splits, seed, device, threshold, guard=None, steps=200):
    for parameter in student.base.parameters():
        parameter.requires_grad_(True)
    if student.cayley:
        student.skew_coordinates.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.base.parameters(), lr=.0001, weight_decay=.01)
    x, y = splits["recovery"]
    indices = minibatches(seed + 60000, len(x), steps, device, batch_size=16)
    history = []
    synchronize(device)
    start = time.perf_counter()
    initial = quality(student, splits["tuning"], teacher)
    hit = {"step": 0, "seconds": time.perf_counter() - start} if initial["perplexity"] <= threshold else None
    checkpoint = copy.deepcopy(student).eval() if hit else None
    history.append({"step": 0, **initial})
    for step, ix in enumerate(indices, 1):
        if guard:
            guard.check()
        optimizer.zero_grad(set_to_none=True)
        logits = student(x[ix], ste=True)
        loss = F.cross_entropy(logits.reshape(-1, 32), y[ix].reshape(-1))
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite recovery loss")
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            metrics = quality(student, splits["tuning"], teacher)
            synchronize(device)
            elapsed = time.perf_counter() - start
            history.append({"step": step, "seconds": elapsed, **metrics})
            if hit is None and metrics["perplexity"] <= threshold:
                hit = {"step": step, "seconds": elapsed}
                checkpoint = copy.deepcopy(student).eval()
    synchronize(device)
    return checkpoint or copy.deepcopy(student).eval(), history, hit, time.perf_counter() - start


def packed_weights(student):
    packed = {}
    with torch.no_grad():
        q = student.rotation()
        for name in QUANTIZED:
            original = getattr(student.base, name).weight
            weight = q @ original if name in ("output", "ff2") else original @ q.T
            scales = weight.abs().amax(1) / 7
            safe = torch.where(scales > 0, scales, torch.ones_like(scales))
            codes = (weight / safe[:, None]).round().clamp(-7, 7).to(torch.int8).add(7).to(torch.uint8).flatten()
            if len(codes) % 2:
                codes = torch.cat((codes, codes.new_zeros(1)))
            packed[name] = {"packed_uint8": codes[0::2] | (codes[1::2] << 4),
                            "scales": scales, "shape": tuple(weight.shape)}
    return packed


def run_seed(out, seed, args):
    directory = out / f"seed-{seed}"
    directory.mkdir()
    splits = sequence_data(seed, args.device, include_test=not args.pilot)
    teacher = ToyTransformer(seed, args.device).eval()
    guard = ResourceGuard(args.device, max_seconds=900)
    synchronize(args.device)
    begin = time.perf_counter()
    teacher_history = pretrain(teacher, splits["recovery"], seed, args.device, steps=0 if args.pilot else 500, guard=guard)
    synchronize(args.device)
    teacher_seconds = time.perf_counter() - begin
    teacher.eval()
    torch.save(teacher.state_dict(), directory / "teacher.pt")
    write_json(directory / "teacher_history.json", teacher_history)
    dense = quality(teacher, splits["tuning"])
    write_json(directory / "dense_tuning.json", dense)
    write_json(directory / "inputs.json", {name: {"x": tensor_hash(x), "y": tensor_hash(y), "count": len(x)}
                                           for name, (x, y) in splits.items()})
    absorbed = absorb_affine(teacher)
    with torch.no_grad():
        hidden = teacher.input_hidden(splits["calibration"][0]).reshape(-1, 64)
    complement = mean_complement(args.device)
    rows, checkpoints = [], {}
    for method in (("identity", "haar") if args.pilot else METHODS):
        method_dir = directory / method
        method_dir.mkdir()
        guard = ResourceGuard(args.device, max_seconds=900)
        row = {"seed": seed, "method": method, "hypothesis": "H03", "status": "running", "stop_reason": ""}
        rows.append(row)
        try:
            synchronize(args.device)
            begin = time.perf_counter()
            q = fixed_rotation(method, hidden, seed, complement)
            student = RotatedQuantized(absorbed, q, cayley=method == "cayley")
            row.update(rotation_checks(student, teacher, splits["calibration"][0][:16]))
            synchronize(args.device)
            row["setup_seconds"] = time.perf_counter() - begin
            row["teacher_shared_seconds"] = teacher_seconds
            row["base_parameters_after_affine_absorption"] = sum(p.numel() for p in student.base.parameters())
            row["extra_cayley_coordinates"] = student.skew_coordinates.numel() if student.cayley else 0
            begin = time.perf_counter()
            calibration = calibrate(student, teacher, splits["calibration"], seed, args.device,
                                    steps=20 if args.pilot else 50, guard=guard)
            synchronize(args.device)
            row["calibration_seconds"] = time.perf_counter() - begin
            checkpoint, history, hit, seconds = recover(student, teacher, splits, seed, args.device,
                dense["perplexity"] * 1.02, guard=guard, steps=0 if args.pilot else 200)
            row.update(recovery_seconds=seconds, quality_reached=hit is not None,
                       recovery_step_to_quality=hit["step"] if hit else None,
                       total_seconds_to_quality=row["setup_seconds"] + row["calibration_seconds"] + hit["seconds"] if hit else None,
                       censored_at_steps=200 if hit is None and not args.pilot else None,
                       status="pilot_complete" if args.pilot else "complete")
            row.update(guard.metrics())
            torch.save(checkpoint.state_dict(), method_dir / "checkpoint.pt")
            packed = packed_weights(checkpoint)
            torch.save(packed, method_dir / "packed_int4.pt")
            row["raw_packed_weight_and_scale_bytes"] = sum(item["packed_uint8"].numel() + 4 * item["scales"].numel() for item in packed.values())
            row["checkpoint_sha256"] = file_hash(method_dir / "checkpoint.pt")
            write_json(method_dir / "calibration_history.json", calibration)
            write_json(method_dir / "recovery_history.json", history)
            checkpoints[method] = checkpoint
        except (ArithmeticError, AssertionError, MemoryError, RuntimeError, TimeoutError, ValueError) as error:
            row.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
            write_json(method_dir / "failure.json", {"row": row, "traceback": traceback.format_exc()})
        write_csv(directory / "rotation_checks.csv", rows)
    for row in rows:
        if row["method"] not in checkpoints:
            continue
        metrics = quality(checkpoints[row["method"]], splits["tuning"] if args.pilot else splits["test"], teacher)
        write_json(directory / row["method"] / "final_metrics.json", metrics)
        row.update(test_perplexity=metrics["perplexity"], logits_normalized_mse=metrics["logits_normalized_mse"],
                   data_split="tuning" if args.pilot else "test")
    if not args.pilot:
        write_json(directory / "dense_final_test.json", quality(teacher, splits["test"]))
    write_csv(directory / "rotation_checks.csv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    args = parser.parse_args(argv)
    args.seeds = (11,) if args.pilot else tuple(int(seed) for seed in args.seeds.split(","))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(SEEDS):
        parser.error("seeds must be distinct protocol seeds")
    torch.set_num_threads(4)
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], check=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__), Path(__file__).with_name("run_h02_synthetic.py"), Path(__file__).with_name("H03_rotations_recovery.md"))
    for path in files:
        shutil.copy2(path, args.out / (path.name + ".source"))
    manifest = {"hypothesis": "H03", "status": "admission", "git_sha": git("rev-parse", "HEAD"),
                "base_sha": BASE_SHA, "command": sys.argv, "source_hashes": {path.name: file_hash(path) for path in files},
                "torch": torch.__version__, "device": args.device, "dtype": "FP32", "seeds": args.seeds,
                "teacher_steps": 0 if args.pilot else 500, "calibration_steps": 20 if args.pilot else 50,
                "recovery_steps": 0 if args.pilot else 200, "teacher_initialization": "N(0,0.02^2); head copied from embedding",
                "dense_teacher_weight_decay": .01, "recovery_weight_decay": .01, "calibration_weight_decay": 0,
                "threshold": "tuning perplexity <= 1.02 * dense tuning perplexity",
                "whole_hypothesis_outcome": "indeterminate",
                "limitations": ["toy learned-position LayerNorm Transformer only; GPT-2/WikiText-2 not run",
                    "absorbed LN affine creates input projection biases equally in all methods; original heads stay unchanged",
                    "all six attention/FFN weight matrices use fake int4; biases/embeddings/head do not",
                    "Cayley adds1953trainable coordinates during calibration only; identity is its fixed-A=0 ablation",
                    "fixed-Q methods calibrate the same absorbed block parameters for50steps",
                    "Cayley frozen during200step recovery; all base model parameters recover with same batches",
                    "raw packed weight/scales bytes exclude serialization and other model parameters",
                    "quantized execution is dequantized FP32; no int4 hardware speed claim",
                    "RMSNorm/RoPE neural control series remain separate; pilot teacher is untrained",
                    "earliest qualifying tuning checkpoint fixed before final test; otherwise final checkpoint censored"]}
    write_json(args.out / "manifest.json", manifest)
    rows = []
    try:
        manifest["admission"] = admission()
        write_json(args.out / "manifest.json", manifest)
        if not args.admission_only:
            for seed in args.seeds:
                rows.extend(run_seed(args.out, seed, args))
                write_csv(args.out / "runs.csv", rows)
                print(json.dumps({"seed": seed, "status": [row["status"] for row in rows if row["seed"] == seed]}), flush=True)
        manifest["status"] = "admission_passed" if args.admission_only else ("complete" if all(row["status"] in ("complete", "pilot_complete") for row in rows) else "partial")
    except BaseException as error:
        manifest.update(status="interrupted", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)
    return 0 if manifest["status"] in ("admission_passed", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
