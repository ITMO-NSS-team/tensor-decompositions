"""Small reproducible CPU protocol; historical results remain unchanged.

Run from the repository after installing tdecomp:
python experiments/two_sided_run.py --shape 48 32 --shape 32 48 --rank 4 --true-rank 12
"""
from argparse import ArgumentParser, SUPPRESS
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import os
import subprocess
import sys
from time import perf_counter, sleep
import tracemalloc

import numpy as np
import torch
import tensorly as tl

from tdecomp.api import SVDContractError, SVDMethod, SVDRequest, compute_svd


@dataclass(frozen=True)
class Protocol:
    shapes: tuple[tuple[int, int], ...] = ((48, 32), (32, 48))
    rank: int = 4
    true_rank: int = 12
    spectrum: str = "geometric"
    noise: float = 0.0
    seed: int = 17
    repeats: int = 3
    warmup: int = 1
    methods: tuple[SVDMethod, ...] = tuple(SVDMethod)
    isolated_memory: bool = False
    worker_timeout_seconds: float = 30.0


def _validate(config):
    if not isinstance(config, Protocol):
        raise ValueError("Expected Protocol")
    for name in ("rank", "true_rank", "seed", "repeats", "warmup"):
        value = getattr(config, name)
        if type(value) is not int or value < (1 if name in ("rank", "repeats") else 0):
            raise ValueError(f"Invalid {name}")
    if not config.shapes or not config.methods or len(set(config.methods)) != len(config.methods):
        raise ValueError("At least one shape and distinct method are required")
    if any(not isinstance(method, SVDMethod) for method in config.methods):
        raise ValueError("Methods must be SVDMethod members")
    if config.spectrum not in ("flat", "geometric") or not np.isfinite(config.noise) or config.noise < 0:
        raise ValueError("Invalid spectrum or noise")
    if type(config.isolated_memory) is not bool or not 0 < config.worker_timeout_seconds <= 60:
        raise ValueError("Invalid isolated-memory mode or timeout (maximum 60 seconds)")
    for shape in config.shapes:
        if (not isinstance(shape, tuple) or len(shape) != 2
                or any(type(d) is not int or d < 1 for d in shape)
                or config.rank > min(shape) or config.true_rank > min(shape)):
            raise ValueError("Invalid shape, target rank, or true rank")
        # Keep this executable protocol bounded independently of API budgets.
        if np.prod(shape) > 2_000_000:
            raise ValueError("Protocol shape exceeds two million entries")


def generate_matrix(shape, true_rank, spectrum, noise, seed):
    """Use a local RNG; true rank, target rank, spectrum, and noise are independent."""
    rng = np.random.default_rng(seed)
    left = np.linalg.qr(rng.normal(size=(shape[0], true_rank)), mode="reduced")[0]
    right = np.linalg.qr(rng.normal(size=(shape[1], true_rank)), mode="reduced")[0]
    singular = np.ones(true_rank) if spectrum == "flat" else np.geomspace(1, 0.01, true_rank)
    X = (left * singular) @ right.T
    if noise:
        X += noise * rng.normal(size=shape) / np.sqrt(np.prod(shape))
    return X


def _provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    digest = sha256()
    sources = sorted((root / "tdecomp").rglob("*.py")) + [Path(__file__).resolve(), root / "pyproject.toml"]
    for source in sources:
        digest.update(str(source.relative_to(root)).replace("\\", "/").encode())
        digest.update(source.read_bytes())
    return {
        "revision": revision, "dirty": dirty, "source_sha256": digest.hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "tensorly": tl.__version__,
        "device": "cpu", "dtype": "float64", "torch_threads": torch.get_num_threads(),
        "timing": "factorization separate from reconstruction; total includes API validation and diagnostics",
        "memory": "tracemalloc Python allocations only; native BLAS/LAPACK peak not measured",
    }


def _execute(X, options, warmup, executor):
    for _ in range(warmup):
        executor(X, options)
    try:
        tracemalloc.start()
        start = perf_counter()
        result = executor(X, options)
        total = perf_counter() - start
        _, peak_python = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    start = perf_counter()
    reconstructed = result.reconstruct()
    reconstruction_seconds = perf_counter() - start
    denominator = np.linalg.norm(X)
    relative_error = 0.0 if denominator == 0 else float(np.linalg.norm(X - reconstructed) / denominator)
    return {"status": "success", "components": result.components,
            "numerical_rank": result.numerical_rank, "relative_error": relative_error,
            "factorization_seconds": result.diagnostics.factorization_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "total_api_seconds": total, "peak_python_traced_bytes": peak_python,
            "peak_native_bytes": None,
            "estimated_working_bytes": result.diagnostics.estimated_working_bytes}


def _memory_worker(spec):
    import psutil
    start = perf_counter()
    X = generate_matrix(tuple(spec["shape"]), spec["true_rank"], spec["spectrum"], spec["noise"], spec["data_seed"])
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    result = _execute(X, SVDRequest(spec["rank"], method=SVDMethod(spec["method"]), seed=spec["method_seed"]), spec["warmup"], compute_svd)
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)
    peak_source = "windows_peak_working_set" if peak is not None else "sampled_process_rss"
    if platform.system() in ("Linux", "Darwin"):
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(value if platform.system() == "Darwin" else value * 1024)
        peak_source = "os_process_peak_rss"
    result.update(worker_seconds=perf_counter() - start,
                  baseline_process_rss_bytes=baseline, os_peak_process_rss_bytes=peak,
                  os_peak_source=peak_source,
                  input_sha256=sha256(X.tobytes()).hexdigest())
    return result


def _isolated_measurement(row, config):
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("--isolated-memory requires tdecomp[experiments] (psutil)") from error
    spec = {key: row[key] for key in ("shape", "true_rank", "spectrum", "noise", "data_seed", "method", "method_seed")}
    spec.update(rank=config.rank, warmup=config.warmup)
    start = perf_counter()
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec", json.dumps(spec)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    observed = psutil.Process(process.pid)
    peak = 0
    interval = 0.005
    try:
        while process.poll() is None:
            if perf_counter() - start > config.worker_timeout_seconds:
                process.kill()
                raise RuntimeError("Isolated CPU worker exceeded its deadline")
            try:
                peak = max(peak, observed.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            sleep(interval)
        stdout, stderr = process.communicate(timeout=1)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
    elapsed = perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Isolated worker failed ({process.returncode}): {stderr[-1000:]}")
    result = json.loads(stdout)
    if result["input_sha256"] != row["input_sha256"]:
        raise RuntimeError("Isolated worker input checksum differs from shared protocol input")
    peak = max(peak, result["os_peak_process_rss_bytes"] or 0)
    result.update(peak_process_rss_bytes=peak,
                  process_rss_peak_above_baseline_bytes=max(0, peak - result["baseline_process_rss_bytes"]),
                  rss_sample_interval_seconds=interval, isolated_total_seconds=elapsed,
                  startup_and_transport_seconds=max(0.0, elapsed - result["worker_seconds"]),
                  memory_scope="fresh process including imports, input, warmup, factorization, reconstruction; not tensor workspace")
    return result


def run_protocol(config=Protocol(), *, executor=compute_svd):
    """Keep one status per configuration/repeat, including explicit failures; no retry."""
    _validate(config)
    rows = []
    for shape_index, shape in enumerate(config.shapes):
        for repetition in range(config.repeats):
            data_seed = config.seed + shape_index * 100_000 + repetition
            X = generate_matrix(shape, config.true_rank, config.spectrum, config.noise, data_seed)
            checksum = sha256(X.tobytes()).hexdigest()
            # Same-rank independent optimum; outside every method timer.
            singular = np.linalg.svd(X, compute_uv=False)
            denominator = np.linalg.norm(X)
            optimal_error = 0.0 if denominator == 0 else float(np.linalg.norm(singular[config.rank:]) / denominator)
            for method in config.methods:
                options = SVDRequest(config.rank, method=method, seed=data_seed + 1_000_000)
                row = {"shape": list(shape), "repeat": repetition, "method": method.value,
                       "target_rank": config.rank, "true_rank": config.true_rank,
                       "spectrum": config.spectrum, "noise": config.noise,
                       "data_seed": data_seed, "method_seed": options.seed,
                       "input_sha256": checksum, "optimal_relative_error": optimal_error}
                try:
                    if config.isolated_memory:
                        if executor is not compute_svd:
                            raise ValueError("Custom executor is supported only in-process")
                        row.update(_isolated_measurement(row, config))
                    else:
                        row.update(_execute(X, options, config.warmup, executor))
                except (SVDContractError, np.linalg.LinAlgError, RuntimeError, MemoryError) as error:
                    row.update(status="failure", error_type=type(error).__name__,
                               error_code=getattr(error, "code", "execution_failure"), error=str(error))
                finally:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                rows.append(row)
    return {"protocol_version": 1, "provenance": _provenance(), "warmup": config.warmup,
            "memory_mode": "isolated_process_rss" if config.isolated_memory else "in_process_python_allocations",
            "expected_records": len(config.shapes) * config.repeats * len(config.methods), "records": rows}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", help=SUPPRESS)
    parser.add_argument("--shape", type=int, nargs=2, action="append")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--true-rank", type=int, default=12)
    parser.add_argument("--spectrum", choices=("flat", "geometric"), default="geometric")
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--method", choices=[method.value for method in SVDMethod], action="append")
    parser.add_argument("--isolated-memory", action="store_true")
    parser.add_argument("--worker-timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/cpu-protocol.json"))
    args = parser.parse_args()
    if args.worker_spec:
        print(json.dumps(_memory_worker(json.loads(args.worker_spec)), allow_nan=False))
        return 0
    config = Protocol(shapes=tuple(tuple(shape) for shape in args.shape) if args.shape else Protocol.shapes,
                      rank=args.rank, true_rank=args.true_rank, spectrum=args.spectrum,
                      noise=args.noise, seed=args.seed, repeats=args.repeats, warmup=args.warmup,
                      methods=tuple(SVDMethod(method) for method in args.method) if args.method else tuple(SVDMethod),
                      isolated_memory=args.isolated_memory, worker_timeout_seconds=args.worker_timeout)
    result = run_protocol(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    failures = sum(row["status"] == "failure" for row in result["records"])
    print(f"{len(result['records'])} records, {failures} failures: {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
