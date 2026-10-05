"""H02 synthetic adjacent-layer comparison with the protocol's nonlinear graph.

The equal-budget comparison is independent versus joint calibration, both from
the same separate SVD initialization. A tied common-Q graph is an explicitly
smaller-parameter ablation. This entry point never reads CIFAR-10. A pilot does
not generate or evaluate the final synthetic test split.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import psutil
import tensorly as tl
import torch
from torch import nn
from torch.nn import functional as F
from tdecomp.matrix.decomposer import SVDDecomposition

SEEDS = (11, 22, 33, 44, 55)
BASE_SHA = "a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b"
METHODS = ("independent", "joint", "shared_q")
SPLIT_COUNTS = (("recovery", 4096), ("calibration", 512), ("tuning", 512), ("test", 1024))


def weight_svd(weight, rank):
    # Explicit context makes standalone execution independent of TensorLy's
    # default NumPy backend, while restoring the caller's backend afterwards.
    with tl.backend_context("pytorch"):
        return SVDDecomposition(rank=rank).decompose(weight)


def activation(x, name):
    if name == "relu":
        return F.relu(x)
    if name == "identity":
        return x
    if name == "silu":
        return F.silu(x)
    raise ValueError(f"unsupported activation: {name}")


class DenseResidualMLP(nn.Module):
    def __init__(self, w1, w2, head, nonlinearity="relu"):
        super().__init__()
        self.w1 = nn.Parameter(w1.clone(), requires_grad=False)
        self.w2 = nn.Parameter(w2.clone(), requires_grad=False)
        self.head = nn.Parameter(head.clone(), requires_grad=False)
        self.nonlinearity = nonlinearity

    def branch(self, x):
        return F.linear(activation(F.linear(x, self.w1), self.nonlinearity), self.w2)

    def block(self, x):
        return x + self.branch(x)

    def forward(self, x):
        return F.linear(self.block(x), self.head)


class AdjacentFactors(nn.Module):
    """A1 -> Q1 -> full-width activation -> Q2.T -> A2 -> residual/head."""
    def __init__(self, a1, q1, q2, a2, head, nonlinearity="relu", tied=False):
        super().__init__()
        self.a1 = nn.Parameter(a1.clone())
        self.q1 = nn.Parameter(q1.clone())
        self.q2 = self.q1 if tied else nn.Parameter(q2.clone())
        self.a2 = nn.Parameter(a2.clone())
        self.register_buffer("head", head.clone())
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.rank = a1.shape[0]
        if (a1.shape[1] != a2.shape[0] or q1.shape != q2.shape
                or q1.shape[1] != self.rank or a2.shape[1] != self.rank):
            raise ValueError("inconsistent adjacent-layer factor shapes")
        if tied and not torch.equal(q1, q2):
            raise ValueError("tied Q initialization must be identical")

    def first_layer(self, x):
        return F.linear(F.linear(x, self.a1), self.q1)

    def second_layer(self, full_width_activation):
        return F.linear(F.linear(full_width_activation, self.q2.T), self.a2)

    def branch(self, x):
        full_width = self.first_layer(x)
        return self.second_layer(activation(full_width, self.nonlinearity))

    def block(self, x):
        return x + self.branch(x)

    def forward(self, x):
        return F.linear(self.block(x), self.head)

    @property
    def factor_parameters(self):
        return tuple(self.parameters())


def initialize(teacher, rank, method):
    """Separate weight SVD start; shared-Q uses the joint hidden energy SVD.

    The ablation's Q is the leading left basis of [W1, W2.T]. Its A factors
    are the corresponding orthogonal projections, and it uses joint fitting.
    This declared initialization is not an extra equal-budget main method.
    """
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if not 1 <= rank <= min(teacher.w1.shape):
        raise ValueError("rank must be between 1 and the smaller weight dimension")
    with torch.no_grad():
        if method == "shared_q":
            q, _, _ = weight_svd(torch.cat((teacher.w1, teacher.w2.T), dim=1), rank)
            a1, a2 = q.T @ teacher.w1, teacher.w2 @ q
            return AdjacentFactors(a1, q, q, a2, teacher.head,
                                   teacher.nonlinearity, tied=True)
        q1, s1, v1 = weight_svd(teacher.w1, rank)
        u2, s2, v2 = weight_svd(teacher.w2, rank)
        return AdjacentFactors(s1[:, None] * v1, q1, v2.T, u2 * s2,
                               teacher.head, teacher.nonlinearity)


def synthetic_data(seed, device="cpu", *, nonlinearity="relu", control="aligned",
                   dtype=torch.float32, include_test=True):
    if control not in ("aligned", "rotated", "flat"):
        raise ValueError(f"unsupported control: {control}")
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    u_full = torch.linalg.qr(torch.randn(64, 64, generator=gen, device=device, dtype=dtype)).Q
    u = u_full[:, :32]
    v = torch.linalg.qr(torch.randn(32, 32, generator=gen, device=device, dtype=dtype)).Q
    v2 = torch.linalg.qr(torch.randn(32, 32, generator=gen, device=device, dtype=dtype)).Q
    right_hidden = u.clone()
    if control == "rotated":
        # Rotate the leading eight directions into the orthogonal complement
        # of the entire 32-dimensional U subspace, preserving orthonormality.
        right_hidden[:, :8] = u_full[:, 32:40]
    spectrum = torch.ones(32, device=device, dtype=dtype)
    if control != "flat":
        spectrum[8:] = 0.3
    w1 = (u * spectrum) @ v.T
    w2 = (v2 * spectrum) @ right_hidden.T
    head = torch.randn(4, 32, generator=gen, device=device, dtype=dtype) / math.sqrt(32)
    teacher = DenseResidualMLP(w1, w2, head, nonlinearity).eval()
    scale = torch.tensor([math.sqrt(2)] * 8 + [math.sqrt(0.5)] * 24,
                         device=device, dtype=dtype)
    splits = {}
    for k, (name, count) in enumerate(SPLIT_COUNTS, 1):
        if name == "test" and not include_test:
            continue
        rng = torch.Generator(device=device).manual_seed(seed + 1000 * k)
        x = torch.randn(count, 32, generator=rng, device=device, dtype=dtype) * scale
        with torch.no_grad():
            splits[name] = (x, teacher(x))
    recipe = {"control": control, "activation": nonlinearity,
              "u": tensor_hash(u), "right_hidden": tensor_hash(right_hidden),
              "leading_overlap_frobenius_squared": float((u[:, :8].T @ right_hidden[:, :8]).square().sum()),
              "spectrum": spectrum.cpu().tolist(), "input_variances": scale.square().cpu().tolist()}
    return teacher, splits, recipe


def minibatches(seed, count, steps, device, batch_size=32):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(count, (steps, batch_size), generator=gen, device=device)


def normalized_error(predicted, exact):
    energy = exact.square().sum()
    residual = (predicted - exact).square().sum()
    if energy.item() == 0:
        return float(residual / exact.numel()), "absolute_mse_zero_target"
    return float(residual / energy), "normalized_squared_error"


@torch.no_grad()
def measure(student, teacher, pair):
    x, y = pair
    block_error, block_kind = normalized_error(student.block(x), teacher.block(x))
    branch_error, _ = normalized_error(student.branch(x), teacher.branch(x))
    prediction = student(x)
    network_error, network_kind = normalized_error(prediction, y)
    return {"block_normalized_mse": block_error, "branch_normalized_mse": branch_error,
            "network_normalized_mse": network_error, "block_metric_kind": block_kind,
            "network_metric_kind": network_kind,
            "teacher_argmax_agreement": float((prediction.argmax(1) == y.argmax(1)).float().mean())}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class ResourceGuard:
    def __init__(self, device, max_seconds=600):
        self.device = torch.device(device)
        # PyTorch 2.2 mem_get_info rejects an unindexed torch.device("cuda").
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.start = time.perf_counter()
        self.max_seconds = max_seconds
        self.peak_ram_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def check(self):
        rss = psutil.Process().memory_info().rss
        self.peak_ram_bytes = max(rss, self.peak_ram_bytes)
        if rss > 24 * 1024**3:
            raise MemoryError("24 GiB process RSS limit reached")
        if time.perf_counter() - self.start > self.max_seconds:
            raise TimeoutError("per-variant time limit reached")
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            if torch.cuda.memory_reserved(self.device) > 12 * 1024**3 or total - free > 12 * 1024**3:
                raise MemoryError("12 GiB device-usage/reserved limit reached")

    def metrics(self):
        self.check()
        return {"peak_ram_bytes": self.peak_ram_bytes,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
                "driver_process_gpu_memory": "unavailable_wddm" if self.device.type == "cuda" else "not_applicable_cpu"}


def train_phase(student, teacher, pair, indices, *, method, phase, guard=None):
    """128 independent updates split 64+64; joint uses the same 128 batches.

    Separate fitting uses teacher inputs to the second layer, avoiding hidden
    contamination from the first compressed layer. Recovery fits teacher
    network outputs using a fresh, identical AdamW policy for every method.
    """
    if method not in METHODS or phase not in ("calibration", "recovery"):
        raise ValueError("unsupported method or training phase")
    if phase == "calibration" and method == "independent" and len(indices) % 2:
        raise ValueError("independent calibration needs an even update count")
    x, y = pair
    optimizer = torch.optim.AdamW(student.factor_parameters, lr=0.001, weight_decay=0)
    losses = []
    targets = []
    half = len(indices) // 2
    for step, ix in enumerate(indices):
        if guard is not None:
            guard.check()
        optimizer.zero_grad(set_to_none=True)
        if phase == "calibration" and method == "independent":
            with torch.no_grad():
                first_exact = F.linear(x[ix], teacher.w1)
            if step < half:
                loss = F.mse_loss(student.first_layer(x[ix]), first_exact)
                target = "first_layer_output"
            else:
                hidden = activation(first_exact, teacher.nonlinearity)
                with torch.no_grad():
                    second_exact = F.linear(hidden, teacher.w2)
                loss = F.mse_loss(student.second_layer(hidden), second_exact)
                target = "second_layer_output_on_teacher_hidden"
        elif phase == "calibration":
            with torch.no_grad():
                exact = teacher.block(x[ix])
            loss = F.mse_loss(student.block(x[ix]), exact)
            target = "block_output"
        else:
            loss = F.mse_loss(student(x[ix]), y[ix])
            target = "network_output"
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
            raise ArithmeticError("nonfinite training gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in student.parameters()):
            raise ArithmeticError("nonfinite trained parameter")
        losses.append(float(loss.detach()))
        targets.append(target)
    return {"steps": len(indices), "minibatches_sha256": tensor_hash(indices),
            "losses": losses, "targets": targets,
            "optimizer": "fresh AdamW(lr=0.001, weight_decay=0)"}


def admission_checks():
    """Independent FP64 graph, gradient, SVD and negative-control checks."""
    gen = torch.Generator().manual_seed(1973)
    w1 = torch.randn(64, 32, generator=gen, dtype=torch.float64)
    w2 = torch.randn(32, 64, generator=gen, dtype=torch.float64)
    head = torch.randn(4, 32, generator=gen, dtype=torch.float64)
    x = torch.randn(7, 32, generator=gen, dtype=torch.float64, requires_grad=True)
    results = {}
    for name in ("identity", "relu", "silu"):
        teacher = DenseResidualMLP(w1, w2, head, name)
        eye = torch.eye(64, dtype=torch.float64)
        lifted = AdjacentFactors(w1, eye, eye, w2, head, name)
        torch.testing.assert_close(lifted(x), teacher(x), atol=1e-10, rtol=1e-10)
        g_dense = torch.autograd.grad(teacher(x).square().mean(), x)[0]
        g_lift = torch.autograd.grad(lifted(x).square().mean(), x)[0]
        torch.testing.assert_close(g_lift, g_dense, atol=1e-9, rtol=1e-10)
        # Compare A1/A2 gradients against independent dense autograd weights.
        ref_w1 = w1.clone().requires_grad_()
        ref_w2 = w2.clone().requires_grad_()
        reference = F.linear(x + F.linear(activation(F.linear(x, ref_w1), name), ref_w2), head)
        ref_grad = torch.autograd.grad(reference.square().mean(), (ref_w1, ref_w2))
        factor_grad = torch.autograd.grad(lifted(x).square().mean(), (lifted.a1, lifted.a2))
        for actual, exact in zip(factor_grad, ref_grad):
            torch.testing.assert_close(actual, exact, atol=1e-9, rtol=1e-10)
        results[f"full_width_identity_graph_{name}"] = "passed_output_and_gradients"
    teacher = DenseResidualMLP(w1, w2, head)
    student = initialize(teacher, 8, "independent")
    u1, s1, vh1 = torch.linalg.svd(w1, full_matrices=False)
    u2, s2, vh2 = torch.linalg.svd(w2, full_matrices=False)
    torch.testing.assert_close(student.q1 @ student.a1, (u1[:, :8] * s1[:8]) @ vh1[:8], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(student.a2 @ student.q2.T, (u2[:, :8] * s2[:8]) @ vh2[:8], atol=1e-10, rtol=1e-10)
    assert sum(p.numel() for p in student.parameters()) == 192 * 8
    tied = initialize(teacher, 8, "shared_q")
    assert tied.q1 is tied.q2 and sum(p.numel() for p in tied.parameters()) == 128 * 8
    results["separate_svd_reference_and_parameter_counts"] = "passed"
    q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64) / math.sqrt(2)
    z = torch.ones(1, 1, dtype=torch.float64)
    moved = F.linear(F.relu(z), q)
    proper = F.relu(F.linear(z, q))
    discrepancy = float((proper - moved).norm())
    assert discrepancy > 0.7
    results["relu_movement_counterexample_norm"] = discrepancy
    return results


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


@torch.no_grad()
def latency(student, x, *, warmups=30, repeats=200):
    if repeats < 1 or warmups < 0:
        raise ValueError("latency requires nonnegative warmups and positive repeats")
    for _ in range(warmups):
        student(x)
    synchronize(x.device)
    times = []
    for _ in range(repeats):
        synchronize(x.device)
        start = time.perf_counter()
        student(x)
        synchronize(x.device)
        times.append(1000 * (time.perf_counter() - start))
    values = torch.tensor(times, dtype=torch.float64)
    return {"p50_ms": float(values.quantile(0.5)), "p95_ms": float(values.quantile(0.95)),
            "warmups": warmups, "repeats": repeats, "batch_size": len(x)}


def paired_summary(rows, expected_seeds=SEEDS):
    complete = {(row["seed"], row["method"]): row for row in rows if row["status"] == "complete"}
    seeds = [seed for seed in expected_seeds if (seed, "joint") in complete and (seed, "independent") in complete]
    values = []
    for seed in seeds:
        base = complete[(seed, "independent")]["block_nmse_before_recovery"]
        joint = complete[(seed, "joint")]["block_nmse_before_recovery"]
        if base > 0:
            values.append((seed, 1 - joint / base))
    n = len(values)
    mean = sum(value for _, value in values) / n if n else None
    low = high = None
    # These are the two-sided 95% Student critical values for the protocol's
    # n <= 5 independent runs. Timing repeats never enter n.
    t95 = {2: 12.7062047364, 3: 4.30265272975, 4: 3.18244630528, 5: 2.7764451052}
    if n in t95:
        variance = sum((value - mean)**2 for _, value in values) / (n - 1)
        margin = t95[n] * math.sqrt(variance / n)
        low, high = mean - margin, mean + margin
    outcome = "indeterminate"
    if n == len(SEEDS) and set(seeds) == set(SEEDS) and low is not None:
        if low >= 0.10:
            outcome = "supported_primary_synthetic_only"
        elif high < 0.10:
            outcome = "minimum_primary_effect_not_supported_in_synthetic_regime"
    return [{"method": "joint", "reference": "independent", "n_seeds": n,
             "mean_delta": mean, "ci95_low": low, "ci95_high": high,
             "metric": "relative_block_error_reduction_before_recovery",
             "criterion": "ci95_low >= 0.10 with all five preregistered seeds",
             "paired_values": json.dumps(values), "outcome": outcome,
             "assumption": "approximately normal paired differences across independent runs"}]


def run_seed(out, seed, args):
    seed_dir = out / f"seed-{seed}"
    seed_dir.mkdir()
    setup_start = time.perf_counter()
    teacher, splits, recipe = synthetic_data(seed, args.device, nonlinearity=args.activation,
                                            control=args.control, include_test=args.mode == "confirm")
    torch.save(teacher.state_dict(), seed_dir / "teacher.pt")
    write_json(seed_dir / "inputs.json", {name: {"count": len(x), "x_sha256": tensor_hash(x),
                                                  "y_sha256": tensor_hash(y)} for name, (x, y) in splits.items()})
    recipe["common_setup_seconds"] = time.perf_counter() - setup_start
    write_json(seed_dir / "recipe.json", recipe)
    cal_steps = 128 if args.mode == "confirm" else 20
    recovery_steps = 512 if args.mode == "confirm" else 0
    batches = {"calibration": minibatches(seed + 50000, 512, cal_steps, args.device),
               "recovery": minibatches(seed + 60000, 4096, recovery_steps, args.device)}
    frozen = {}
    rows = []
    for method in args.methods:
        method_dir = seed_dir / method
        method_dir.mkdir()
        guard = ResourceGuard(args.device, args.max_seconds)
        row = {"hypothesis": "H02", "stage": args.mode, "seed": seed, "method": method,
               "rank_tuple": str(args.rank), "sketch_size": "not_applicable", "data_split": "tuning",
               "n_examples": 512, "checkpoint_sha": "", "status": "running", "stop_reason": "",
               "equal_budget_main_comparison": method != "shared_q",
               "calibration_steps": cal_steps, "recovery_steps": recovery_steps, "batch_size": 32}
        rows.append(row)
        try:
            synchronize(args.device)
            start = time.perf_counter()
            student = initialize(teacher, args.rank, method)
            synchronize(args.device)
            row["factor_seconds"] = time.perf_counter() - start
            row["factor_parameters"] = sum(p.numel() for p in student.parameters())
            row["frozen_head_parameters"] = teacher.head.numel()
            row["setup_seconds"] = 0.0
            histories = {}
            checkpoints = {}
            tuning = {"initial": measure(student, teacher, splits["tuning"])}
            for phase in ("calibration", "recovery"):
                synchronize(args.device)
                start = time.perf_counter()
                histories[phase] = train_phase(student, teacher, splits[phase], batches[phase],
                                                method=method, phase=phase, guard=guard)
                synchronize(args.device)
                row[f"{phase}_seconds"] = time.perf_counter() - start
                tuning[phase] = measure(student, teacher, splits["tuning"])
                checkpoints[phase] = copy.deepcopy(student).eval()
                torch.save(student.state_dict(), method_dir / f"{phase}.pt")
                write_json(method_dir / "history.json", histories)
                write_json(method_dir / "tuning.json", tuning)
            row["checkpoint_sha"] = file_hash(method_dir / "recovery.pt")
            row["calibration_checkpoint_sha"] = file_hash(method_dir / "calibration.pt")
            row["block_nmse_before_recovery_tuning"] = tuning["calibration"]["block_normalized_mse"]
            row["network_nmse_after_recovery_tuning"] = tuning["recovery"]["network_normalized_mse"]
            row["total_seconds"] = row["factor_seconds"] + row["calibration_seconds"] + row["recovery_seconds"]
            row["method_wall_seconds_including_reporting"] = time.perf_counter() - guard.start
            row.update(guard.metrics())
            row["status"] = "complete" if args.mode == "confirm" else "pilot_complete"
            frozen[method] = checkpoints
            write_json(method_dir / "status.json", row)
        except (ArithmeticError, AssertionError, MemoryError, RuntimeError, TimeoutError, ValueError) as error:
            row.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
            write_json(method_dir / "failure.json", {"status": row, "traceback": traceback.format_exc()})
        write_csv(out / "runs.partial.csv", rows)
    # Every checkpoint is frozen and saved before this single final test pass.
    # Primary pre-recovery and secondary post-recovery checkpoints are evaluated
    # together; no test metric changes training, rank or checkpoint selection.
    if args.mode == "confirm":
        for row in rows:
            if row["status"] != "complete":
                continue
            checkpoints = frozen[row["method"]]
            metrics = {phase: measure(model, teacher, splits["test"]) for phase, model in checkpoints.items()}
            write_json(seed_dir / row["method"] / "final_test.json", metrics)
            row.update(data_split="test", n_examples=1024,
                       primary_metric=metrics["calibration"]["block_normalized_mse"],
                       secondary_metric=metrics["recovery"]["network_normalized_mse"],
                       block_nmse_before_recovery=metrics["calibration"]["block_normalized_mse"],
                       block_nmse_after_recovery=metrics["recovery"]["block_normalized_mse"],
                       network_nmse_after_recovery=metrics["recovery"]["network_normalized_mse"],
                       teacher_argmax_agreement=metrics["recovery"]["teacher_argmax_agreement"])
    timing_rows = []
    warmups, repeats = (30, 200) if args.mode == "confirm" else (3, 5)
    for method, model in [("dense", teacher)] + [(method, phases["recovery"]) for method, phases in frozen.items()]:
        for count in (1, 32):
            measured = latency(model, splits["tuning"][0][:count], warmups=warmups, repeats=repeats)
            timing_rows.append({"seed": seed, "method": method, **measured})
            if count == 1:
                for row in rows:
                    if row["method"] == method and row["status"] in ("complete", "pilot_complete"):
                        row.update(p50_ms=measured["p50_ms"], p95_ms=measured["p95_ms"])
    write_csv(seed_dir / "latency.csv", timing_rows)
    if args.activation == "identity":
        u, s, vh = weight_svd(teacher.w2 @ teacher.w1, args.rank)
        composition = (u * s) @ vh
        pair = splits["test"] if args.mode == "confirm" else splits["tuning"]
        x, _ = pair
        oracle_block = x + F.linear(x, composition)
        error, kind = normalized_error(oracle_block, teacher.block(x))
        write_json(seed_dir / "linear_oracle.json", {"method": "SVD_W2_W1_identity_only",
                    "block_normalized_mse": error, "metric_kind": kind,
                    "data_split": "test" if args.mode == "confirm" else "tuning",
                    "equal_budget_main_comparison": False})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "confirm"), default="pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--methods", help="comma-separated methods; pilot allows only one")
    parser.add_argument("--rank", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--activation", choices=("relu", "identity", "silu"), default="relu")
    parser.add_argument("--control", choices=("aligned", "rotated", "flat"), default="aligned")
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--base-sha", default=BASE_SHA)
    args = parser.parse_args(argv)
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    args.methods = tuple(args.methods.split(",")) if args.methods else (METHODS if args.mode == "confirm" else ("joint",))
    if not args.methods or len(set(args.methods)) != len(args.methods) or not set(args.methods).issubset(METHODS):
        parser.error("methods must be distinct supported method names")
    if args.mode == "pilot" and len(args.methods) != 1:
        parser.error("the protocol pilot allows one method only")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("seeds must be distinct and nonempty")
    if args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if args.mode == "confirm" and not set(args.seeds).issubset(SEEDS):
        parser.error("confirmation seeds must come from the five preregistered seeds")
    if args.mode == "pilot" and len(args.seeds) != 1:
        # Default confirmation seeds must not silently multiply the pilot.
        if args.seeds == SEEDS:
            args.seeds = (SEEDS[0],)
        else:
            parser.error("the protocol pilot allows one initialization only")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        # The variable must be set before CUDA context creation; check rather
        # than silently switching deterministic policy during an experiment.
        import os
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before a CUDA run")
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", args.base_sha, "HEAD"], check=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = Path(__file__).with_name("H02_adjacent_subspaces.md")
    for path in (Path(__file__).resolve(), protocol):
        shutil.copy2(path, out / f"{path.name}.source")
    versions = {}
    for package in ("torch", "torchvision", "tensorly", "numpy", "psutil"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    manifest = {"hypothesis": "H02", "mode": args.mode, "status": "running",
                "command": sys.argv, "config": {**vars(args), "out": str(out)},
                "git_sha": git("rev-parse", "HEAD"), "base_sha": args.base_sha,
                "git_status": git("status", "--short"), "python": platform.python_version(),
                "versions": versions, "compiled_cuda": torch.version.cuda, "dtype": "torch.float32",
                "source_sha256": file_hash(__file__), "protocol_sha256": file_hash(protocol),
                "split_counts": dict(SPLIT_COUNTS), "primary_relative_reduction_threshold": 0.10,
                "admission_checks": "pending", "whole_hypothesis_outcome": "indeterminate",
                "scope": "synthetic regression; no real-model evaluation or full H02 conclusion",
                "nonlinearity_position": "after Q1 lift into the original 64-wide hidden space",
                "shared_q": {"factor_parameters": "128r", "main_factor_parameters": "192r",
                             "equal_budget_control": False, "calibration": "joint block objective"},
                "time_to_quality": "not_reported: no synthetic regression quality threshold specified",
                "argmax_agreement": "secondary teacher agreement; not observed classification accuracy",
                "common_setup_cost": "recorded once per seed; excluded from relative method construction cost",
                "timing_policy": "report batch 1 and 32; 30 warmups/200 repeats in confirm, 3/5 in pilot",
                "test_policy": "not generated in pilot; evaluated only on frozen saved checkpoints in confirm"}
    write_json(out / "manifest.json", manifest)
    write_json(out / "units.json", {"mse": "dimensionless normalized energy ratio except zero-target absolute MSE",
                  "seconds": "seconds", "p50_ms": "milliseconds", "p95_ms": "milliseconds",
                  "memory": "bytes", "teacher_argmax_agreement": "fraction of predictions matching teacher argmax",
                  "mean_delta": "relative block error reduction (1 - joint/independent)"})
    rows = []
    try:
        manifest["admission_checks"] = admission_checks()
        write_json(out / "manifest.json", manifest)
        for seed in args.seeds:
            rows.extend(run_seed(out, seed, args))
            write_csv(out / "runs.csv", rows)
            write_csv(out / "paired_summary.csv", paired_summary(rows))
            print(json.dumps({"seed": seed, "status": [row["status"] for row in rows if row["seed"] == seed]}), flush=True)
        manifest["status"] = "complete" if all(row["status"] in ("complete", "pilot_complete") for row in rows) else "partial"
        manifest["primary_synthetic_summary"] = paired_summary(rows)
    except BaseException as error:
        manifest.update(status="interrupted", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
