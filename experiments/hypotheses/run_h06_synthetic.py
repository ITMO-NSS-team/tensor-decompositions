"""H06 finite-grid rank growth, direct certificates and equal CNN recovery.

The adaptive rule scores measured residual-energy reduction per extra stored
coefficient. It only grows on the declared grid and falls back to full rank.
No stochastic probe is promoted to a certificate or a signal-rank estimator.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from experiments.hypotheses import synthetic_tucker_common as common
from experiments.hypotheses.run_h02_synthetic import (
    BASE_SHA, ResourceGuard, file_hash, git, latency, synchronize, tensor_hash, write_csv, write_json,
)

GRID = ((2, 4, 8), (2, 3, 6), (1, 2), (1, 2))
METHODS = ("hosvd_equal", "st_equal", "fractional", "adaptive")


def validate_order(order, ndim=4):
    if sorted(order) != list(range(ndim)):
        raise ValueError("order must contain each tensor mode exactly once")


def stored_coefficients(shape, ranks):
    import math
    return math.prod(ranks) + sum(n * r for n, r in zip(shape, ranks))


def st_hosvd(tensor, ranks, order=(0, 1, 2, 3)):
    validate_order(order, tensor.ndim)
    core = tensor
    factors = [None] * tensor.ndim
    discarded = []
    for mode in order:
        u, _, _ = common.left_svd(common.unfold(core, mode),ranks[mode])
        factors[mode] = u[:, :ranks[mode]]
        projected = common.mode_dot(core, factors[mode].T, mode)
        # Measure actual sequential discarded energy, not another unfolding's
        # singular-value tail. Keep the direct difference for cancellation.
        lifted = common.mode_dot(projected, factors[mode], mode)
        discarded.append(float((core - lifted).square().sum()))
        core = projected
    return core, factors, discarded


def full_rank(tensor):
    return tensor.clone(), [torch.eye(n, device=tensor.device, dtype=tensor.dtype) for n in tensor.shape]


def certificate(tensor, core, factors, epsilon=.10, *, discarded=None):
    reconstructed = common.reconstruct(core, factors)
    relative, absolute = common.relative_residual(tensor, reconstructed)
    energy = float(tensor.square().sum())
    difference = energy - float(core.square().sum())
    direct_squared = float((tensor - reconstructed).square().sum())
    tolerance = 128 * torch.finfo(tensor.dtype).eps * max(energy, 1e-30)
    orthogonal_error = max(float((factor.T @ factor - torch.eye(factor.shape[1], device=factor.device,
                                  dtype=factor.dtype)).norm()) for factor in factors)
    if orthogonal_error > (1e-5 if tensor.dtype == torch.float32 else 1e-10):
        raise ArithmeticError("certificate factors are not orthonormal")
    if abs(difference - direct_squared) > tolerance:
        raise ArithmeticError("projection energy identity disagrees with direct residual")
    near_cancellation = abs(difference) <= tolerance
    fp64_direct = None
    if near_cancellation:
        precise = common.reconstruct(core.double(), [factor.double() for factor in factors])
        fp64_direct = float((tensor.double() - precise).square().sum())
    return {"relative_residual": relative, "absolute_residual": absolute,
            "zero_tensor_relative_error_undefined": energy == 0,
            "admissible": absolute == 0 if energy == 0 else relative <= epsilon,
            "epsilon": epsilon, "energy_difference_unclamped": difference,
            "direct_residual_energy": direct_squared, "near_cancellation": near_cancellation,
            "direct_fp64_residual_energy": fp64_direct, "orthogonality_error": orthogonal_error,
            "sequential_discarded_energy": discarded,
            "sequential_energy_identity_error": abs(sum(discarded) - direct_squared) if discarded is not None else None,
            "certificate_scope": "stored orthogonal projection before neural recovery"}


def probe_diagnostics(tensor, factors, seed):
    results = []
    for mode, factor in enumerate(factors):
        matrix = common.unfold(tensor, mode)
        residual = matrix - factor @ (factor.T @ matrix)
        gen = torch.Generator(device=tensor.device).manual_seed(seed + 90000 + mode)
        probes = torch.randn(matrix.shape[1], 16, generator=gen, device=tensor.device, dtype=tensor.dtype)
        estimate = float((residual @ probes).square().sum() / 16)
        results.append({"mode": mode, "probes": 16, "seed": seed + 90000 + mode,
                        "estimated_modal_residual_energy": estimate,
                        "actual_modal_residual_energy": float(residual.square().sum()), "certificate": False})
    return results


def equal_budget(tensor, *, sequential, epsilon=.10, order=(0, 1, 2, 3)):
    """Each mode receives epsilon^2 ||T||^2 / 4; missing grid rank -> full."""
    validate_order(order)
    threshold = epsilon**2 * float(tensor.square().sum()) / tensor.ndim
    current = tensor
    factors = [None] * tensor.ndim
    ranks = [None] * tensor.ndim
    discarded = []
    for mode in order:
        matrix = common.unfold(current if sequential else tensor, mode)
        u, s, _ = common.left_svd(matrix)
        chosen = next((rank for rank in GRID[mode] if float(s[rank:].square().sum()) <= threshold), None)
        if chosen is None:
            core, full_factors = full_rank(tensor)
            return core, full_factors, {"ranks": tuple(tensor.shape), "full_rank_fallback": True,
                                        "fallback_reason": "equal modal budget has no permitted grid rank",
                                        "sequential_discarded_energy": None}
        ranks[mode] = chosen
        factors[mode] = u[:, :chosen]
        if sequential:
            projected = common.mode_dot(current, factors[mode].T, mode)
            lifted = common.mode_dot(projected, factors[mode], mode)
            discarded.append(float((current - lifted).square().sum()))
            current = projected
    core = current if sequential else common.project(tensor, factors)
    info = {"ranks": tuple(ranks), "full_rank_fallback": False,
            "sequential_discarded_energy": discarded if sequential else None,
            "modal_budget_absolute_squared": threshold}
    return core, factors, info


def adaptive(tensor, *, epsilon=.10, order=(0, 1, 2, 3), guard=None):
    validate_order(order)
    cache = {}
    trace = []

    def evaluate(ranks):
        if ranks not in cache:
            if guard:
                guard.check()
            start = time.perf_counter()
            core, factors, discarded = st_hosvd(tensor, ranks, order)
            evidence = certificate(tensor, core, factors, epsilon, discarded=discarded)
            cache[ranks] = core, factors, evidence
            trace.append({"ranks": ranks, "parameters": stored_coefficients(tensor.shape, ranks),
                          "relative_residual": evidence["relative_residual"],
                          "residual_energy": evidence["direct_residual_energy"],
                          "seconds": time.perf_counter() - start, "selected_next": False})
        return cache[ranks]

    current = tuple(values[0] for values in GRID)
    while True:
        core, factors, evidence = evaluate(current)
        if evidence["admissible"]:
            return core, factors, {"ranks": current, "full_rank_fallback": False,
                                   "search_trace": trace, "searched_tuples": len(cache)}
        candidates = []
        current_count = stored_coefficients(tensor.shape, current)
        for mode, values in enumerate(GRID):
            index = values.index(current[mode])
            if index + 1 < len(values):
                ranks = list(current)
                ranks[mode] = values[index + 1]
                ranks = tuple(ranks)
                _, _, new_evidence = evaluate(ranks)
                additional = stored_coefficients(tensor.shape, ranks) - current_count
                benefit = evidence["direct_residual_energy"] - new_evidence["direct_residual_energy"]
                candidates.append((benefit / additional, -additional, -mode, ranks))
        if not candidates:
            core, factors = full_rank(tensor)
            return core, factors, {"ranks": tuple(tensor.shape), "full_rank_fallback": True,
                "fallback_reason": "finite grid exhausted before direct residual admission",
                "search_trace": trace, "searched_tuples": len(cache)}
        current = max(candidates)[3]
        for entry in trace:
            if tuple(entry["ranks"]) == current:
                entry["selected_next"] = True


def choose(tensor, method, *, epsilon=.10, fraction=.5, order=(0, 1, 2, 3), guard=None):
    synchronize(tensor.device)
    start = time.perf_counter()
    if method in ("hosvd_equal", "st_equal"):
        core, factors, info = equal_budget(tensor, sequential=method == "st_equal", epsilon=epsilon, order=order)
    elif method == "adaptive":
        core, factors, info = adaptive(tensor, epsilon=epsilon, order=order, guard=guard)
    elif method == "fractional":
        import math
        ranks = tuple(max(1, min(n, math.ceil(n * fraction))) for n in tensor.shape)
        core, factors = common.exact_hosvd(tensor, ranks)
        info = {"ranks": ranks, "full_rank_fallback": False, "fraction": fraction}
    else:
        raise ValueError("unsupported H06 rank rule")
    evidence = certificate(tensor, core, factors, epsilon, discarded=info.get("sequential_discarded_energy"))
    if not evidence["admissible"]:
        # A fractional candidate may be inadmissible; preserve that fact and
        # take the explicitly costed full-rank reserve for neural comparison.
        info["inadmissible_candidate"] = {"ranks": info["ranks"], "certificate": evidence}
        core, factors = full_rank(tensor)
        info.update(ranks=tuple(tensor.shape), full_rank_fallback=True)
        evidence = certificate(tensor, core, factors, epsilon)
    synchronize(tensor.device)
    info.update(factor_seconds=time.perf_counter() - start, certificate=evidence,
                factor_parameters=common.parameter_count(core, factors), mode_order=order)
    return core, factors, info


def admission():
    result = common.common_admission()
    for scale in (1e-6, 1., 1e6):
        tensor, _, _ = common.signal_weight(11, dtype=torch.float64, scale=scale)
        for order in ((0, 1, 2, 3), (2, 3, 1, 0)):
            core, factors, info = choose(tensor, "adaptive", order=order)
            assert info["certificate"]["admissible"]
            assert info["searched_tuples"] <= 36
    zero = torch.zeros(common.SHAPE, dtype=torch.float64)
    core, factors, info = choose(zero, "adaptive")
    assert info["ranks"] == (2, 2, 1, 1)
    assert info["certificate"]["relative_residual"] is None
    assert info["certificate"]["absolute_residual"] == 0
    full_core, full_factors = full_rank(zero)
    assert certificate(zero, full_core, full_factors)["energy_difference_unclamped"] == 0
    result.update(adaptive_direct_certificate_scales_and_orders="passed", bounded_grid="passed",
                  zero_relative_error_undefined="passed", full_rank_reserve="passed")
    return result


def run_seed(out, seed, args):
    directory = out / f"seed-{seed}"
    directory.mkdir()
    teacher, splits, signal = common.synthetic_cnn(seed, args.device, include_test=not args.pilot,
        sigma=args.sigma, flat=args.control == "flat", scale=args.scale, zero=args.control == "zero")
    torch.save(teacher.state_dict(), directory / "teacher.pt")
    write_json(directory / "inputs.json", {name: {"count": len(x), "x": tensor_hash(x), "y": tensor_hash(y)}
                                           for name, (x, y) in splits.items()})
    tensor = teacher[2].weight.detach()
    rows, frozen = [], {}
    for method in args.methods:
        method_dir = directory / method
        method_dir.mkdir()
        guard = ResourceGuard(args.device)
        row = {"hypothesis": "H06", "seed": seed, "method": method, "status": "running", "stop_reason": ""}
        rows.append(row)
        try:
            core, factors, info = choose(tensor, method, epsilon=.10, fraction=args.fraction,
                                         order=args.order, guard=guard)
            write_json(method_dir / "rank_selection.json", info)
            row.update(rank_tuple=str(info["ranks"]), factor_parameters=info["factor_parameters"],
                       factor_seconds=info["factor_seconds"], relative_residual=info["certificate"]["relative_residual"],
                       safety_violation=not info["certificate"]["admissible"], full_rank_fallback=info["full_rank_fallback"])
            row["signal_relative_residual"], _ = common.relative_residual(signal, common.reconstruct(core, factors))
            diagnostic_start = time.perf_counter()
            write_json(method_dir / "probes.json", probe_diagnostics(tensor, factors, seed))
            synchronize(args.device)
            row["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
            student = common.student_from(teacher, core, factors)
            write_json(method_dir / "tuning_before.json", common.cnn_metrics(student, teacher, splits["tuning"]))
            history, timing = common.recover_cnn(student, splits, seed, args.device, pilot=args.pilot, guard=guard)
            row.update(timing)
            row["full_cost_seconds"] = row["factor_seconds"] + row["diagnostic_seconds"] + sum(timing.values())
            row.update(guard.metrics())
            torch.save(student.state_dict(), method_dir / "checkpoint.pt")
            write_json(method_dir / "history.json", history)
            row.update(checkpoint_sha256=file_hash(method_dir / "checkpoint.pt"),
                       status="pilot_complete" if args.pilot else "complete")
            row["variant_wall_seconds_before_final_test"] = time.perf_counter() - guard.start
            frozen[method] = student
        except (ArithmeticError, AssertionError, MemoryError, RuntimeError, TimeoutError, ValueError) as error:
            row.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
            write_json(method_dir / "failure.json", {"row": row, "traceback": traceback.format_exc()})
        write_csv(directory / "results.csv", rows)
    for row in rows:
        if row["method"] not in frozen:
            continue
        pair = splits["tuning"] if args.pilot else splits["test"]
        metrics = common.cnn_metrics(frozen[row["method"]], teacher, pair)
        row.update(network_mse=metrics["mse"], network_normalized_mse=metrics["normalized_mse"],
                   teacher_argmax_agreement=metrics["teacher_argmax_agreement"], data_split="tuning" if args.pilot else "test")
        write_json(directory / row["method"] / "metrics.json", metrics)
        row.update(latency(frozen[row["method"]], splits["tuning"][0][:1],
                            warmups=3 if args.pilot else 30, repeats=5 if args.pilot else 200))
    write_csv(directory / "results.csv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--fraction", type=float, choices=(.25, .5, .75), default=.5)
    parser.add_argument("--order", choices=("0,1,2,3", "2,3,1,0"), default="0,1,2,3")
    parser.add_argument("--sigma", type=float, choices=(0., .05, .2), default=.05)
    parser.add_argument("--control", choices=("signal", "flat", "zero"), default="signal")
    parser.add_argument("--scale", type=float, choices=(1e-6, 1., 1e6), default=1.)
    args = parser.parse_args(argv)
    args.seeds = (11,) if args.pilot else tuple(int(seed) for seed in args.seeds.split(","))
    args.order = tuple(int(mode) for mode in args.order.split(","))
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(common.SEEDS):
        parser.error("seeds must be distinct protocol seeds")
    args.methods = ("adaptive",) if args.pilot else METHODS
    torch.set_num_threads(4)
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], check=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__), Path(common.__file__), Path(__file__).with_name("run_h02_synthetic.py"),
             Path(__file__).with_name("H06_residual_rank_budget.md"))
    for path in files:
        shutil.copy2(path, args.out / (path.name + ".source"))
    manifest = {"hypothesis": "H06", "status": "admission", "git_sha": git("rev-parse", "HEAD"),
                "base_sha": BASE_SHA, "source_hashes": {path.name: file_hash(path) for path in files},
                "command": sys.argv, "torch": torch.__version__, "device": args.device, "dtype": "FP32",
                "seeds": args.seeds, "grid": GRID, "epsilon": .10, "fraction": args.fraction,
                "order": args.order, "control": args.control, "scale": args.scale, "sigma": args.sigma,
                "whole_hypothesis_outcome": "indeterminate", "growth_rule": "actual residual-energy gain / extra stored coefficient",
                "limitations": ["observed-weight certificate before recovery; not signal rank or network-quality guarantee",
                    "orthogonality not imposed during neural recovery; certificate does not extend to recovered factors",
                    "dense temporary reconstructed convolution; actual execution cost included",
                    "all trial tuples included in search time; cached decompositions reused",
                    "fraction .5 fixed by default; .25/.75 and alternate order are explicit additional controls",
                    "16 Gaussian probes per mode are diagnostics only; direct full residual is the certificate",
                    "full-rank fallback may store more coefficients than original dense weight",
                    "real-model confirmation and paired interval interpretation remain separate"]}
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "units.json", {"relative_residual": "dimensionless norm ratio",
        "energy": "squared Frobenius norm", "factor_parameters": "stored scalar coefficients",
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
    return 0 if manifest["status"] in ("admission_passed", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
