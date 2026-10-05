"""Exact and TensorSketch factor LS, plus equal-recovery H05 synthetic CNN.

TensorSketch acts on the composite other-mode row index. Both A and B receive
the same operator. TUCKER-TTMTS is not implemented or claimed. Exact local LS
validation is included in the sketch method's measured construction cost.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

import tensorly as tl
import torch
from tdecomp.tensor.tucker import HOOIDecomposition
from experiments.hypotheses import synthetic_tucker_common as common
from experiments.hypotheses.run_h02_synthetic import (
    BASE_SHA, ResourceGuard, file_hash, git, latency, synchronize, tensor_hash,
    write_csv, write_json,
)

METHODS = ("exact_als", "refreshed_tensorsketch", "fixed_tensorsketch", "hooi")


@dataclass(frozen=True)
class TensorSketch:
    dimensions: tuple
    rows: int
    hashes: tuple
    signs: tuple

    @classmethod
    def draw(cls, dimensions, rows, seed, device="cpu", dtype=torch.float32):
        if rows < 1 or any(n < 1 for n in dimensions):
            raise ValueError("TensorSketch dimensions and rows must be positive")
        generator = torch.Generator(device=device).manual_seed(seed)
        hashes = tuple(torch.randint(rows, (n,), generator=generator, device=device) for n in dimensions)
        signs = tuple((2 * torch.randint(2, (n,), generator=generator, device=device) - 1).to(dtype) for n in dimensions)
        return cls(tuple(dimensions), rows, hashes, signs)

    def composite(self):
        device = self.hashes[0].device
        hashes = torch.zeros(self.dimensions, dtype=torch.long, device=device)
        signs = torch.ones(self.dimensions, dtype=self.signs[0].dtype, device=device)
        for mode, (h, sign) in enumerate(zip(self.hashes, self.signs)):
            shape = [1] * len(self.dimensions)
            shape[mode] = len(h)
            hashes = hashes + h.reshape(shape)
            signs = signs * sign.reshape(shape)
        return hashes.reshape(-1).remainder(self.rows), signs.reshape(-1)

    def apply(self, matrix):
        if matrix.shape[0] != math.prod(self.dimensions):
            raise ValueError("TensorSketch row index shape mismatch")
        hashes, signs = self.composite()
        result = matrix.new_zeros(self.rows, matrix.shape[1])
        return result.index_add(0, hashes, matrix * signs[:, None])

    def explicit(self):
        hashes, signs = self.composite()
        result = signs.new_zeros(self.rows, len(signs))
        result[hashes, torch.arange(len(signs), device=signs.device)] = signs
        return result

    def kron_fft(self, factors):
        if tuple(factor.shape[0] for factor in factors) != self.dimensions:
            raise ValueError("Kronecker factor row dimensions disagree with sketch")
        frequency = None
        for factor, hashes, signs in zip(factors, self.hashes, self.signs):
            projected = factor.new_zeros(self.rows, factor.shape[1]).index_add(0, hashes, factor * signs[:, None])
            spectrum = torch.fft.fft(projected, dim=0)
            frequency = spectrum if frequency is None else (frequency[:, :, None] * spectrum[:, None, :]).reshape(self.rows, -1)
        # torch's inverse FFT includes 1/s, exactly the circular convolution
        # normalization; no extra division or sqrt(s) scaling is introduced.
        return torch.fft.ifft(frequency, dim=0).real


def svd_ls(a, b, rcond=1e-6):
    if a.shape[0] != b.shape[0]:
        raise ValueError("LS operators must have the same row count")
    u, singular, vh = torch.linalg.svd(a, full_matrices=False)
    inverse = torch.zeros_like(singular)
    if len(singular):
        keep = singular > rcond * singular.max()
        inverse[keep] = singular[keep].reciprocal()
    return (vh.T * inverse) @ (u.T @ b)


def local_guard(a, b, z, exact_z, epsilon=0.2):
    exact = float((a @ exact_z - b).norm())
    actual = float((a @ z - b).norm())
    rounding_floor = 64 * torch.finfo(a.dtype).eps * max(float(b.norm()), 1e-30)
    bound = math.sqrt((1 + epsilon) / (1 - epsilon)) * exact + rounding_floor
    if not math.isfinite(actual) or actual > bound:
        raise ArithmeticError(f"true LS residual exceeds local admission bound: {actual} > {bound}")
    return actual, exact


def ls_operators(tensor, core, factors, mode):
    partial = core
    for other, factor in enumerate(factors):
        if other != mode:
            partial = common.mode_dot(partial, factor, other)
    return common.unfold(partial, mode).T, common.unfold(tensor, mode).T


def factor_qr(core, factors, mode, raw_factor):
    before_factors = list(factors)
    before_factors[mode] = raw_factor
    before = common.reconstruct(core, before_factors)
    q, r = torch.linalg.qr(raw_factor, mode="reduced")
    transported = common.mode_dot(core, r, mode)
    after_factors = list(factors)
    after_factors[mode] = q
    after = common.reconstruct(transported, after_factors)
    error = float((before - after).norm()) / max(float(before.norm()), 1e-30)
    tolerance = 1e-5 if core.dtype == torch.float32 else 1e-10
    if error > tolerance:
        raise ArithmeticError(f"QR/core transport changed reconstruction: {error}")
    return transported, after_factors, error


def factorize(tensor, ranks, method, seed, *, sketch_rows=32, max_sweeps=10, tolerance=1e-5, guard=None):
    if method not in METHODS:
        raise ValueError("unsupported H05 method")
    if not 1 <= max_sweeps <= 10:
        raise ValueError("the H05 protocol allows at most ten sweeps")
    synchronize(tensor.device)
    start = time.perf_counter()
    core, factors = common.exact_hosvd(tensor, ranks)
    initial_hashes = {"core": tensor_hash(core), "factors": [tensor_hash(f) for f in factors]}
    if method == "hooi":
        with tl.backend_context("pytorch"):
            core, factors = HOOIDecomposition(rank=ranks, random_state=seed).decompose(
                tensor, init=(core, factors), n_iter_max=max_sweeps, tol=tolerance)
        synchronize(tensor.device)
        return core, factors, [], {"factor_seconds": time.perf_counter() - start,
                                  "initialization_hashes": initial_hashes, "algorithm": "orthogonal HOOI"}
    fixed = {}
    if method == "fixed_tensorsketch":
        for mode in range(tensor.ndim):
            dimensions = tuple(n for k, n in enumerate(tensor.shape) if k != mode)
            fixed[mode] = TensorSketch.draw(dimensions, sketch_rows, seed + 70000 + mode,
                                            tensor.device, tensor.dtype)
    previous, _ = common.relative_residual(tensor, common.reconstruct(core, factors))
    previous = previous or 0.0
    rows = []
    core_projection_seconds = 0.0
    for sweep in range(max_sweeps):
        for mode in range(tensor.ndim):
            if guard:
                guard.check()
            step_start = time.perf_counter()
            a, b = ls_operators(tensor, core, factors, mode)
            exact_z = svd_ls(a, b)
            sketch = None
            if method != "exact_als":
                if sketch_rows < ranks[mode]:
                    raise ValueError("s < modal rank is an inadmissible sketch control")
                dimensions = tuple(n for k, n in enumerate(tensor.shape) if k != mode)
                sketch = fixed.get(mode) or TensorSketch.draw(dimensions, sketch_rows,
                    seed + 80000 + 1000 * sweep + mode, tensor.device, tensor.dtype)
                other_factors = [factor for k, factor in enumerate(factors) if k != mode]
                sa = sketch.kron_fft(other_factors) @ common.unfold(core, mode).T
                sb = sketch.apply(b)
                z = svd_ls(sa, sb)
            else:
                z = exact_z
            true_residual, optimal_residual = local_guard(a, b, z, exact_z)
            sketch_residual = float((sa @ z - sb).norm()) if sketch else None
            core, factors, qr_error = factor_qr(core, factors, mode, z.T)
            singular = torch.linalg.svdvals(a)
            condition = float(singular.max() / singular.min()) if float(singular.min()) > 0 else None
            synchronize(tensor.device)
            rows.append({"seed": seed, "method": method, "sweep": sweep, "mode": mode,
                         "rows": a.shape[0], "cols": a.shape[1], "sketch_rows": sketch_rows if sketch else 0,
                         "refresh_policy": "per_LS" if method == "refreshed_tensorsketch" else ("fixed_per_mode" if sketch else "none"),
                         "cond_a": condition, "ls_true_residual": true_residual,
                         "ls_exact_optimum_residual": optimal_residual, "ls_sketch_residual": sketch_residual,
                         "qr_reconstruction_error": qr_error, "core_policy": "QR transport; exact projection after sweep",
                         "rows_read": a.shape[0], "seconds": time.perf_counter() - step_start,
                         "local_exact_validation_included": True,
                         "sketch_hash": tensor_hash(torch.cat(sketch.hashes)) if sketch else ""})
        projection_start = time.perf_counter()
        core = common.project(tensor, factors)
        synchronize(tensor.device)
        core_projection_seconds += time.perf_counter() - projection_start
        current, _ = common.relative_residual(tensor, common.reconstruct(core, factors))
        current = current or 0.0
        improvement = (previous - current) / max(previous, 1e-30)
        if improvement < tolerance:
            break
        previous = current
    synchronize(tensor.device)
    return core, factors, rows, {"factor_seconds": time.perf_counter() - start,
                                "core_projection_seconds": core_projection_seconds,
                                "initialization_hashes": initial_hashes,
                                "algorithm": "fixed-core factor LS with gauge transport and per-sweep core projection",
                                "local_exact_validation_included": True, "sweeps": sweep + 1}


def admission():
    results = common.common_admission()
    gen = torch.Generator().manual_seed(521)
    factors = [torch.randn(n, r, generator=gen, dtype=torch.float64) for n, r in ((8, 3), (3, 2), (3, 2))]
    kron = torch.kron(torch.kron(factors[0], factors[1]), factors[2])
    for size in (2, 17, 32):
        sketch = TensorSketch.draw((8, 3, 3), size, 15 + size, dtype=torch.float64)
        torch.testing.assert_close(sketch.apply(kron), sketch.explicit() @ kron, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(sketch.kron_fft(factors), sketch.explicit() @ kron, atol=1e-11, rtol=1e-12)
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64)
    core, orthogonal = common.exact_hosvd(tensor, common.SIGNAL_RANKS)
    for mode in range(4):
        a, b = ls_operators(tensor, core, orthogonal, mode)
        other = [factor for k, factor in enumerate(orthogonal) if k != mode]
        product = other[0]
        for factor in other[1:]:
            product = torch.kron(product, factor)
        torch.testing.assert_close(a, product @ common.unfold(core, mode).T, atol=1e-12, rtol=1e-12)
        reference = torch.linalg.lstsq(a, b, rcond=1e-6, driver="gelsd").solution
        torch.testing.assert_close(svd_ls(a, b), reference, atol=1e-11, rtol=1e-10)
        factor_qr(core, orthogonal, mode, reference.T)
    results.update(tensorsketch_explicit_fft_collisions="passed", unfolding_kronecker_order="passed",
                   exact_LS_vs_independent_gelsd="passed", qr_core_transport="passed")
    return results


def run_seed(out, seed, args):
    directory = out / f"seed-{seed}"
    directory.mkdir()
    teacher, splits, signal = common.synthetic_cnn(seed, args.device, include_test=not args.pilot, sigma=args.sigma)
    torch.save(teacher.state_dict(), directory / "teacher.pt")
    write_json(directory / "inputs.json", {name: {"count": len(x), "x": tensor_hash(x), "y": tensor_hash(y)}
                                           for name, (x, y) in splits.items()})
    tensor = teacher[2].weight.detach()
    rows, ls_rows = [], []
    frozen = {}
    for method in args.methods:
        method_dir = directory / method
        method_dir.mkdir()
        guard = ResourceGuard(args.device)
        row = {"seed": seed, "method": method, "hypothesis": "H05", "status": "running",
               "ranks": str(common.SIGNAL_RANKS), "sketch_rows": args.sketch_rows, "stop_reason": ""}
        rows.append(row)
        try:
            core, factors, steps, stats = factorize(tensor, common.SIGNAL_RANKS, method, seed,
                                                    sketch_rows=args.sketch_rows, guard=guard)
            ls_rows.extend(steps)
            row.update(stats)
            row["observed_rel_error"], _ = common.relative_residual(tensor, common.reconstruct(core, factors))
            row["signal_rel_error"], _ = common.relative_residual(signal, common.reconstruct(core, factors))
            row["factor_parameters"] = common.parameter_count(core, factors)
            student = common.student_from(teacher, core, factors)
            write_json(method_dir / "tuning_before.json", common.cnn_metrics(student, teacher, splits["tuning"]))
            history, timing = common.recover_cnn(student, splits, seed, args.device, pilot=args.pilot, guard=guard)
            row.update(timing)
            row["full_cost_seconds"] = row["factor_seconds"] + sum(timing.values())
            row.update(guard.metrics())
            torch.save(student.state_dict(), method_dir / "checkpoint.pt")
            write_json(method_dir / "history.json", history)
            row["checkpoint_sha256"] = file_hash(method_dir / "checkpoint.pt")
            row["status"] = "pilot_complete" if args.pilot else "complete"
            frozen[method] = student
        except (ArithmeticError, AssertionError, MemoryError, RuntimeError, TimeoutError, ValueError) as error:
            row.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
            write_json(method_dir / "failure.json", {"row": row, "traceback": traceback.format_exc()})
        write_csv(directory / "ls_steps.csv", ls_rows)
        write_csv(directory / "tensor_results.csv", rows)
    for row in rows:
        if row["method"] not in frozen:
            continue
        student = frozen[row["method"]]
        pair = splits["tuning"] if args.pilot else splits["test"]
        metrics = common.cnn_metrics(student, teacher, pair)
        row["network_mse"] = metrics["mse"]
        row["network_normalized_mse"] = metrics["normalized_mse"]
        row["teacher_argmax_agreement"] = metrics["teacher_argmax_agreement"]
        row["data_split"] = "tuning" if args.pilot else "test"
        write_json(directory / row["method"] / "metrics.json", metrics)
        row.update(latency(student, splits["tuning"][0][:1], warmups=3 if args.pilot else 30, repeats=5 if args.pilot else 200))
    write_csv(directory / "tensor_results.csv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--sketch-rows", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--sigma", type=float, choices=(0., .05, .2), default=.05)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    args = parser.parse_args(argv)
    args.seeds = (11,) if args.pilot else tuple(int(seed) for seed in args.seeds.split(","))
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(common.SEEDS):
        parser.error("seeds must be distinct protocol seeds")
    args.methods = ("exact_als",) if args.pilot else METHODS
    torch.set_num_threads(4)
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], check=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__), Path(common.__file__), Path(__file__).with_name("run_h02_synthetic.py"),
             Path(__file__).with_name("H05_sketched_tucker_ls.md"))
    for path in files:
        shutil.copy2(path, args.out / (path.name + ".source"))
    manifest = {"hypothesis": "H05", "status": "admission", "git_sha": git("rev-parse", "HEAD"),
                "base_sha": BASE_SHA, "source_hashes": {path.name: file_hash(path) for path in files},
                "command": sys.argv, "torch": torch.__version__, "tensorly": tl.__version__,
                "device": args.device, "dtype": "FP32", "seeds": args.seeds, "sketch_rows": args.sketch_rows,
                "sigma": args.sigma, "methods": args.methods, "whole_hypothesis_outcome": "indeterminate",
                "limitations": ["synthetic only; full hypotheses require real neural comparison",
                    "dense temporary Tucker reconstruction is included in execution; no network speed claim",
                    "local exact-LS validation and true-residual checks included in full construction time",
                    "fixed sketch receives no adaptive-cycle embedding guarantee",
                    "argmax teacher agreement is not observed classification accuracy",
                    "TUCKER-TTMTS not implemented; no global convergence guarantee"]}
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "units.json", {"residuals": "Frobenius norm", "rel_error": "dimensionless norm ratio",
        "seconds": "seconds", "p50_ms": "milliseconds", "p95_ms": "milliseconds", "memory": "bytes"})
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
    return 0 if manifest["status"] in ("complete", "admission_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
