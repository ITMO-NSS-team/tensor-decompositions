"""CPU necessary-feasibility audit of H06's real ResNet18 rank grids.

For every original mode unfolding, Eckart--Young gives an unavoidable
squared residual tail. Their maximum is a lower bound for *any* Tucker
approximation with the specified multilinear ranks. A passing lower bound
does not certify a Tucker approximation or neural quality. This preliminary
audit performs no candidate optimization, model forward pass or data access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
BASE_SHA = "a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b"
SEEDS = (101, 202, 303)
LAYER_SHAPES = {"layer2.1.conv2": (128, 128, 3, 3),
                "layer3.0.conv2": (256, 256, 3, 3)}
EPSILON = 0.10
BASELINE_MIN_ACCURACY = 0.70


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty audit table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


class CPUGuard:
    """RSS/time guard which never queries or initializes CUDA."""
    def __init__(self, seconds=600):
        self.started = time.perf_counter()
        self.max_seconds = seconds
        self.peak_ram_bytes = 0

    def check(self):
        self.peak_ram_bytes = max(self.peak_ram_bytes, psutil.Process().memory_info().rss)
        if self.peak_ram_bytes > 24 * 1024**3:
            raise MemoryError("24 GiB process RSS limit reached")
        if time.perf_counter() - self.started > self.max_seconds:
            raise TimeoutError("600-second CPU preliminary audit budget exhausted")


def real_rank_grid(layer):
    """All allowed ranks, used for exclusion only, not the <=12 search stage."""
    if layer not in LAYER_SHAPES:
        raise ValueError("layer is not one of H06's two prespecified targets")
    channels = (16, 32, 64, 96, 128) if layer == "layer2.1.conv2" else (16, 32, 64, 96)
    result = list(itertools.product(channels, channels, (1, 2, 3), (1, 2, 3)))
    reserve = LAYER_SHAPES[layer]
    if reserve not in result:
        result.append(reserve)
    return tuple(result)


def modal_spectra(tensor, guard=None):
    """LAPACK SVD values of the original unfoldings; no U/Vh allocation.

    FP32 checkpoint entries are converted exactly to FP64. This audit-only
    precision choice is recorded; it is not a changed neural search policy.
    """
    if tensor.device.type != "cpu" or tensor.ndim != 4:
        raise ValueError("a four-mode CPU tensor is required")
    if not torch.isfinite(tensor).all():
        raise ArithmeticError("nonfinite source weight")
    source = tensor.detach().to(dtype=torch.float64)
    energy = float(source.square().sum())
    if not math.isfinite(energy):
        raise ArithmeticError("nonfinite tensor energy")
    modes = []
    for mode in range(4):
        if guard is not None:
            guard.check()
        unfolding = source.movedim(mode, 0).reshape(source.shape[mode], -1)
        started = time.perf_counter()
        singular_values = torch.linalg.svdvals(unfolding)
        elapsed = time.perf_counter() - started
        if not torch.isfinite(singular_values).all():
            raise ArithmeticError("nonfinite singular values")
        spectral_energy = float(singular_values.square().sum())
        error = abs(spectral_energy - energy) / energy if energy else abs(spectral_energy)
        if error > 1e-10:
            raise ArithmeticError("original unfolding spectrum violates Frobenius energy identity")
        tails = [float(singular_values[rank:].square().sum())
                 for rank in range(source.shape[mode] + 1)]
        modes.append({"mode": mode, "unfolding_shape": list(unfolding.shape),
                      "singular_values": singular_values.tolist(), "tail_squared": tails,
                      "spectral_energy": spectral_energy,
                      "relative_energy_identity_error": error, "svd_seconds": elapsed})
        if guard is not None:
            guard.check()
    # A conservative floating-point decision margin, not a rigorous interval.
    margin = 128 * torch.finfo(torch.float64).eps * max(max(m["unfolding_shape"]) for m in modes)
    return {"shape": list(tensor.shape), "source_dtype": str(tensor.dtype),
            "spectrum_dtype": "torch.float64", "source_weight_sha256": tensor_hash(tensor),
            "norm_squared": energy, "decision_margin_relative_squared": margin,
            "modes": modes}


def rank_evidence(spectra, ranks, epsilon=EPSILON):
    shape = tuple(spectra["shape"])
    ranks = tuple(ranks)
    if len(ranks) != 4 or any(type(r) is not int or not 1 <= r <= n for r, n in zip(ranks, shape)):
        raise ValueError("four positive integer ranks bounded by mode sizes are required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    tail_squared = [m["tail_squared"][rank] for m, rank in zip(spectra["modes"], ranks)]
    bound_squared = max(tail_squared)
    energy = spectra["norm_squared"]
    relative_squared = bound_squared / energy if energy else None
    reserve = ranks == shape
    if reserve:
        status = "exact_full_rank_reserve"
    elif relative_squared is None:
        status = "zero_tensor_absolute_error_zero_relative_undefined"
    elif relative_squared > epsilon**2 + spectra["decision_margin_relative_squared"]:
        status = "excluded_by_necessary_lower_bound"
    else:
        status = "not_excluded_not_certified"
    stored = math.prod(ranks) + sum(n * r for n, r in zip(shape, ranks))
    return {"ranks": list(ranks), "modal_tail_squared": tail_squared,
            "unavoidable_residual_squared_lower_bound": bound_squared,
            "relative_error_lower_bound": math.sqrt(relative_squared) if energy else None,
            "epsilon": epsilon, "status": status, "full_rank_reserve": reserve,
            "tucker_coefficients": stored, "dense_coefficients": math.prod(shape),
            "mathematical_storage_ratio": stored / math.prod(shape),
            "mathematical_storage_reduction": stored < math.prod(shape),
            "tucker_candidate_constructed": False}


def baseline_admission(baseline):
    """Check completed train-only baselines using metadata and file hashes."""
    baseline = Path(baseline)
    path = baseline / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("state") != "completed" or manifest.get("seeds") != list(SEEDS)
            or manifest.get("epochs") != 30 or manifest.get("final_test_opened") is not False):
        raise ValueError("baseline must be completed for 101/202/303, 30 epochs, test unopened")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, manifest["git_sha"]],
                   cwd=REPO, check=True)
    result = {"manifest_sha256": file_hash(path), "git_sha": manifest["git_sha"],
              "final_test_opened": False, "data_archive_sha256_from_baseline": manifest["data_archive_sha256"],
              "split_sha256_from_baseline": manifest["split_sha256"],
              "dataset_files_read": False, "baseline_cost_reused_not_remeasured": True,
              "seeds": {}}
    for seed in SEEDS:
        result_path = baseline / f"seed-{seed}" / "result.json"
        item = json.loads(result_path.read_text(encoding="utf-8"))
        if (item.get("state") != "baseline_ready" or item.get("seed") != seed
                or item.get("epochs") != 30 or item.get("final_test_opened") is not False
                or item["final_tuning"]["n"] != 3000
                or not math.isfinite(item["final_tuning"]["accuracy"])
                or item["final_tuning"]["accuracy"] < BASELINE_MIN_ACCURACY):
            raise ValueError(f"baseline seed {seed} does not pass the original tuning admission")
        checkpoint = baseline / f"seed-{seed}" / "model.pt"
        actual_hash = file_hash(checkpoint)
        if actual_hash != item["checkpoint_sha256"]:
            raise ValueError(f"baseline seed {seed} checkpoint hash mismatch")
        result["seeds"][str(seed)] = {"checkpoint_sha256": actual_hash,
                                      "result_sha256": file_hash(result_path),
                                      "original_tuning": item["final_tuning"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    guard = CPUGuard()
    torch.set_num_threads(args.cpu_threads)
    protocol = Path(__file__).with_name("H06_residual_rank_budget.md")
    tests = REPO / "tests" / "test_h06_real_feasibility.py"
    sources = (Path(__file__), protocol, tests)
    source_hashes = {}
    for source in sources:
        source_hashes[str(source.relative_to(REPO))] = file_hash(source)
        shutil.copyfile(source, args.out / (source.name + ".source"))
    manifest = {"state": "admission", "hypothesis": "H06", "setting": "real",
                "stage": "preliminary_original_unfolding_necessary_feasibility",
                "neural_experiment_completed": False, "seeds": list(SEEDS),
                "layers": {name: list(shape) for name, shape in LAYER_SHAPES.items()},
                "epsilon": EPSILON, "base_sha": BASE_SHA, "git_sha": git("rev-parse", "HEAD"),
                "git_status": git("status", "--short"), "source_hashes": source_hashes,
                "command": sys.argv, "device": "cpu", "cpu_threads": args.cpu_threads,
                "torch": torch.__version__, "python": platform.python_version(),
                "package_versions": {name: metadata.version(name) for name in ("torchvision", "tensorly", "psutil")},
                "cuda_runtime_from_build_metadata_only": torch.version.cuda,
                "gpu_driver_query": "not_performed_cpu_only", "gpu_calls_performed": False,
                "final_test_opened": False, "precision": "FP64 exact-spectrum CPU preliminary audit of original FP32 weights",
                "lower_bound_formula": "sqrt(max_k sum_{j>r_k} sigma_j(T_(k))^2 / ||T||_F^2)",
                "limitations": ["necessary lower bounds do not certify any compressed Tucker approximation",
                                "enumeration only excludes impossible tuples; no <=12-tuple method search or neural training is performed",
                                "FP64 LAPACK spectra and floating-point decision margin are numerical evidence, not rigorous interval arithmetic",
                                "mathematical coefficients are not execution memory or speed",
                                "baseline tuning metrics are reused from metadata; no dataset or model forward pass is opened"],
                "limits": {"rss_bytes": 24 * 1024**3, "total_seconds": 600}}
    write_json(args.out / "manifest.json", manifest)
    rows, modal_rows, summaries = [], [], []
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=REPO, check=True)
        manifest["corrected_base_gate"] = "passed_current_and_baseline_ancestry"
        admission = baseline_admission(args.baseline)
        write_json(args.out / "baseline_admission.json", admission)
        shutil.copyfile(args.baseline / "manifest.json", args.out / "baseline-manifest.json.source")
        manifest["baseline_admission_sha256"] = file_hash(args.out / "baseline_admission.json")
        manifest["state"] = "running"
        write_json(args.out / "manifest.json", manifest)
        for seed in SEEDS:
            guard.check()
            checkpoint = args.baseline / f"seed-{seed}" / "model.pt"
            shutil.copyfile(checkpoint.parent / "result.json", args.out / f"baseline-seed-{seed}-result.json.source")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for layer, expected_shape in LAYER_SHAPES.items():
                weight = state[layer + ".weight"]
                if weight.dtype != torch.float32 or tuple(weight.shape) != expected_shape:
                    raise ValueError(f"unexpected original weight for {seed}/{layer}")
                spectra = modal_spectra(weight, guard)
                filename = f"seed-{seed}-{layer}-spectra.json"
                write_json(args.out / filename, spectra)
                evidence = [rank_evidence(spectra, rank) for rank in real_rank_grid(layer)]
                prefix = {"seed": seed, "layer": layer,
                          "checkpoint_sha256": admission["seeds"][str(seed)]["checkpoint_sha256"],
                          "source_weight_sha256": spectra["source_weight_sha256"]}
                for item in evidence:
                    rows.append({**prefix, **{k: json.dumps(v) if isinstance(v, list) else v for k, v in item.items()}})
                for mode in spectra["modes"]:
                    for rank, tail in enumerate(mode["tail_squared"]):
                        modal_rows.append({"seed": seed, "layer": layer, "mode": mode["mode"],
                                           "rank": rank, "tail_squared": tail,
                                           "relative_modal_tail": math.sqrt(tail / spectra["norm_squared"]) if spectra["norm_squared"] else None})
                remaining = [item for item in evidence if item["status"] == "not_excluded_not_certified"]
                summary = {"seed": seed, "layer": layer, "grid_tuples_including_reserve": len(evidence),
                           "excluded_tuples": sum(item["status"] == "excluded_by_necessary_lower_bound" for item in evidence),
                           "not_excluded_compressed_tuples": [item["ranks"] for item in remaining if item["mathematical_storage_reduction"]],
                           "not_excluded_other_nonreserve_tuples": [item["ranks"] for item in remaining if not item["mathematical_storage_reduction"]],
                           "best_nonreserve_relative_lower_bound": min(item["relative_error_lower_bound"] for item in evidence if not item["full_rank_reserve"]),
                           "channel_rank96_modal_lower_bounds": [math.sqrt(spectra["modes"][m]["tail_squared"][96] / spectra["norm_squared"]) for m in (0, 1)],
                           "spatial_rank2_modal_lower_bounds": [math.sqrt(spectra["modes"][m]["tail_squared"][2] / spectra["norm_squared"]) for m in (2, 3)],
                           "spectra_sha256": file_hash(args.out / filename),
                           "full_rank_reserve": list(expected_shape), "full_rank_reserve_exact_error": 0.0,
                           "scope": "necessary feasibility only; no constructed compressed candidate or neural experiment"}
                summaries.append(summary)
                write_json(args.out / "summary.json", summaries)
                write_csv(args.out / "grid_bounds.csv", rows)
                write_csv(args.out / "modal_tails.csv", modal_rows)
                print(json.dumps(summary), flush=True)
            del state
        manifest["state"] = "preliminary_feasibility_completed"
        manifest["summary_sha256"] = file_hash(args.out / "summary.json")
        write_json(args.out / "table_units.json", {"grid_bounds.csv": {"tail_squared": "squared Frobenius weight units", "relative_error_lower_bound": "dimensionless", "mathematical_storage_ratio": "Tucker/dense coefficient count"},
                                                   "modal_tails.csv": {"tail_squared": "squared Frobenius weight units", "relative_modal_tail": "dimensionless"}})
    except Exception as error:
        manifest.update(state="preliminary_audit_failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(total_seconds=time.perf_counter() - guard.started,
                        peak_ram_bytes=guard.peak_ram_bytes, peak_gpu_bytes=0,
                        final_test_opened=False, neural_experiment_completed=False)
        write_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
