"""H12 real two-layer rank selection with shared seed101 search and frozen BN.

Search compares six candidates per rule after32 calibration updates. The
remaining528-update cost is a forecast. Selected searched candidates receive
32+128 calibration and400 recovery updates; the fixed32 on seeds202/303 is not
a new rank search. The forced full-rank reserve receives only128/400, without
extra initial32. Test access follows persistence of all nine checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import tensorly as tl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets, models, transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from run_h04_real import TuckerConv, admission as operator_admission, evaluate
from run_h06_real_feasibility import BASE_SHA, SEEDS, baseline_admission, file_hash, tensor_hash, write_json, write_csv, git
from tdecomp.tensor.tucker import HOOIDecomposition

TARGETS = ("layer2.1.conv2", "layer3.0.conv2")
CHANNELS = (128, 256)
FUNCTIONAL_PAIRS = (((16, 16), (32, 32)), ((32, 32), (32, 32)),
                    ((32, 32), (64, 64)), ((64, 64), (32, 32)),
                    ((64, 64), (64, 64)), ((96, 96), (96, 96)))
FRACTIONS = (.125, .25, .375, .5, .625, .75)
EPSILONS = (.03, .05, .10, .15, .20, .30)
RULES = ("functional", "fractional", "energy")
FULL_PAIR = ((128, 128), (256, 256))
HORIZON = 10000
QUALITY_DROP = .01


class ResourceGuard:
    def __init__(self, device, max_seconds=3600):
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.max_seconds = max_seconds
        self.peak_ram_bytes = 0
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.reset_peak_memory_stats(self.device)

    def check(self):
        self.peak_ram_bytes = max(self.peak_ram_bytes, psutil.Process().memory_info().rss)
        if self.peak_ram_bytes > 24 * 1024**3:
            raise MemoryError("24 GiB process RSS limit reached")
        if time.perf_counter() - self.started > self.max_seconds:
            raise TimeoutError("H12 phase wall-time budget exhausted")
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            if total - free > 12 * 1024**3 or torch.cuda.memory_reserved(self.device) > 12 * 1024**3:
                raise MemoryError("12 GiB total-device/reserved memory limit reached")

    def metrics(self):
        self.check()
        return {"peak_ram_bytes": self.peak_ram_bytes,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def layer_at(model, target):
    group, block, name = target.split(".")
    return getattr(getattr(model, group)[int(block)], name)


def replace_layer(model, target, layer):
    group, block, name = target.split(".")
    setattr(getattr(model, group)[int(block)], name, layer)


def freeze_except_targets(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for target in TARGETS:
        for parameter in layer_at(model, target).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return tuple(trainable)


def storage_count(pair, channels=CHANNELS):
    return sum(n * (out_rank + in_rank) + 9 * out_rank * in_rank
               for n, (out_rank, in_rank) in zip(channels, pair))


def fractional_pair(fraction, channels=CHANNELS):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must belong to (0,1]")
    # H12 real protocol fixes the spatial ranks at3/3 for every rule.
    return tuple((min(n, math.ceil(n * fraction)),) * 2 for n in channels)


def original_channel_spectra(weight, guard=None):
    if weight.ndim != 4 or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
        raise ValueError("finite original FP32 convolution weight required")
    norm_squared = float(weight.square().sum())
    modes = []
    for mode in (0, 1):
        if guard is not None:
            guard.check()
        unfolding = weight.detach().movedim(mode, 0).reshape(weight.shape[mode], -1)
        # cuSOLVER Jacobi FP32 can lose modal energy beyond the fixed guard.
        # Use QR-based SVD for spectra; preserve FP32 and the original tolerance.
        driver = "gesvd" if unfolding.device.type == "cuda" else None
        singular = torch.linalg.svdvals(unfolding, driver=driver)
        if not torch.isfinite(singular).all():
            raise ArithmeticError("nonfinite original modal spectrum")
        identity_error = abs(float(singular.square().sum()) - norm_squared) / norm_squared if norm_squared else 0.0
        if identity_error > 1e-5:
            raise ArithmeticError("FP32 original modal spectrum energy identity failed")
        modes.append({"mode": mode, "unfolding_shape": list(unfolding.shape),
                      "singular_values": singular.cpu().tolist(),
                      "tail_squared": [float(singular[r:].square().sum()) for r in range(weight.shape[mode] + 1)],
                      "relative_energy_identity_error": identity_error})
    return {"shape": list(weight.shape), "norm_squared": norm_squared, "modes": modes,
            "weight_sha256": tensor_hash(weight), "precision": "FP32 original unfoldings",
            "svd_driver": "gesvd" if weight.device.type == "cuda" else "CPU native",
            "spatial_ranks": [3, 3], "spatial_discarded_energy": [0.0, 0.0]}


def energy_ranks(spectra, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    budget = epsilon**2 * spectra["norm_squared"] / 4
    ranks = tuple(next(rank for rank in range(1, size + 1) if mode["tail_squared"][rank] <= budget)
                  for size, mode in zip(spectra["shape"][:2], spectra["modes"]))
    return ranks, {"epsilon": epsilon, "per_mode_squared_budget": budget,
                   "channel_tail_squared": [mode["tail_squared"][rank] for mode, rank in zip(spectra["modes"], ranks)],
                   "rank_rule": "smallest positive original modal rank with tail<=epsilon^2*||T||^2/4; spatial3/3"}


def rule_candidates(rule, spectra=None):
    if rule == "functional":
        return [(pair, {"prespecified_index": index}) for index, pair in enumerate(FUNCTIONAL_PAIRS)]
    if rule == "fractional":
        return [(fractional_pair(f), {"fraction": f}) for f in FRACTIONS]
    if rule == "energy":
        if spectra is None or len(spectra) != 2:
            raise ValueError("two original layer spectra are required")
        result = []
        for epsilon in EPSILONS:
            entries = [energy_ranks(layer, epsilon) for layer in spectra]
            result.append((tuple(entry[0] for entry in entries), {"epsilon": epsilon,
                           "layer_rank_evidence": [entry[1] for entry in entries]}))
        return result
    raise ValueError("unknown H12 selection rule")


def make_candidate(teacher, pair, seed):
    student = copy.deepcopy(teacher)
    residuals = []
    with torch.no_grad(), tl.backend_context("pytorch"):
        for target, (out_rank, in_rank) in zip(TARGETS, pair):
            original = layer_at(teacher, target)
            weight = original.weight.detach()
            if original.bias is not None or original.stride != (1, 1) or original.padding != (1, 1):
                raise ValueError("H04 graph requires the prescribed stride1, padding1, bias-free convolution")
            if not (1 <= out_rank <= weight.shape[0] and 1 <= in_rank <= weight.shape[1]):
                raise ValueError("channel rank outside the original mode size")
            if (out_rank, in_rank) == tuple(weight.shape[:2]):
                # An exact visible safety reserve, with full storage and no
                # extra HOOI numerical error or hidden free dense parameter.
                spatial = weight.clone()
                u = torch.eye(out_rank, dtype=weight.dtype, device=weight.device)
                v = torch.eye(in_rank, dtype=weight.dtype, device=weight.device)
            else:
                core, factors = HOOIDecomposition(rank=(out_rank, in_rank, 3, 3), random_state=seed).decompose(
                    weight, n_iter_max=20, tol=1e-6)
                spatial = tl.tenalg.multi_mode_dot(core, factors[2:], modes=[2, 3])
                u, v = factors[:2]
            replacement = TuckerConv(spatial, u, v)
            reconstructed = replacement.weight()
            # Native FP32 residuals in the real run; tiny CPU admission may
            # supply FP64 tensors independently. No hidden FP64 CUDA policy.
            norm = float(weight.norm())
            direct = float((weight - reconstructed).norm())
            residuals.append({"layer": target, "ranks": [out_rank, in_rank, 3, 3],
                              "original_weight_sha256": tensor_hash(weight),
                              "direct_weight_relative_error": direct / norm if norm else None,
                              "direct_weight_absolute_error": direct,
                              "residual_dtype": str(weight.dtype),
                              "factor_parameter_count": sum(p.numel() for p in replacement.parameters())})
            replace_layer(student, target, replacement)
    trainable = freeze_except_targets(student)
    if sum(p.numel() for p in trainable) != storage_count(pair, tuple(layer_at(teacher, target).weight.shape[0] for target in TARGETS)):
        raise ArithmeticError("actual factor parameter count differs from H12 mathematical count")
    return student, residuals


def constant_batch_indices(count, steps, seed, batch_size=128):
    if count <= 0 or steps < 0:
        raise ValueError("positive split size and nonnegative steps required")
    rng = torch.Generator().manual_seed(seed)
    chunks, total = [], 0
    while total < steps * batch_size:
        permutation = torch.randperm(count, generator=rng)
        chunks.append(permutation)
        total += count
    return torch.cat(chunks)[:steps * batch_size].reshape(steps, batch_size) if steps else torch.empty(0, batch_size, dtype=torch.int64)


def restore_candidate(teacher, state):
    """Restore fixed factor shapes without paying a second HOOI search/setup."""
    student = copy.deepcopy(teacher)
    for target in TARGETS:
        prefix = target + "."
        replace_layer(student, target, TuckerConv(state[prefix + "core"], state[prefix + "u"], state[prefix + "v"]))
    student.load_state_dict(state)
    freeze_except_targets(student)
    return student


def train_updates(model, dataset, split, seed, device, steps, guard, microbatch=128, optimizer=None):
    if microbatch not in (128, 64, 32):
        raise ValueError("microbatch must be128/64/32; effective batch is128")
    parameters = freeze_except_targets(model)
    if optimizer is None:
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    seed_all(seed, device)
    positions = constant_batch_indices(len(split), steps, seed)
    original_indices = torch.as_tensor(split, dtype=torch.int64)[positions]
    synchronize(device)
    started = time.perf_counter()
    history = []
    for step, indices in enumerate(original_indices):
        guard.check()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for chunk in indices.split(microbatch):
            x, y = default_collate([dataset[int(ix)] for ix in chunk])
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x).float(), y) * len(chunk) / 128
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite H12 calibration/recovery loss")
            loss.backward()
            loss_value += float(loss.detach())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ArithmeticError("missing/nonfinite H12 factor gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ArithmeticError("nonfinite H12 trained factor")
        history.append({"step": step, "loss": loss_value, "effective_batch": 128})
    synchronize(device)
    return {"seconds": time.perf_counter() - started, "steps": steps,
            "original_batch_indices_sha256": tensor_hash(original_indices), "microbatch": microbatch,
            "effective_batch": 128, "history": history}, optimizer


@torch.no_grad()
def measured_latency(model, x, device, guard, warmups=30, repeats=200):
    model.eval()
    values = []
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for _ in range(warmups):
            guard.check()
            model(x)
        synchronize(device)
        for _ in range(repeats):
            guard.check()
            synchronize(device)
            started = time.perf_counter()
            model(x)
            synchronize(device)
            values.append(1000 * (time.perf_counter() - started))
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "warmups": warmups, "measured": repeats, "batch": len(x),
            "precision": "BF16 autocast/FP32 parameters" if device == "cuda" else "FP32"}


def choose_index(candidates):
    eligible = [i for i, candidate in enumerate(candidates) if candidate["admissible"]]
    return min(eligible, key=lambda i: (candidates[i]["forecast_cost_without_common_search"],
                                      tuple(tuple(r) for r in candidates[i]["pair"]), candidates[i]["index"])) if eligible else None


def initial_steps(selection):
    return 0 if selection["full_rank_reserve"] else 32


def paired_cost_summary(rows, final):
    """Numeric paired evidence only; no automatic H12 acceptance claim."""
    result = []
    for reference in ("fractional", "energy"):
        values = []
        quality_ok = []
        for seed in SEEDS:
            functional = next(row for row in rows if row["seed"] == seed and row["rule"] == "functional")
            baseline = next(row for row in rows if row["seed"] == seed and row["rule"] == reference)
            values.append(1 - functional["measured_cost_seconds_per_image"] / baseline["measured_cost_seconds_per_image"])
            quality_ok.extend(next(item["quality_passed"] for item in final if item["seed"] == seed and item["rule"] == rule)
                              for rule in ("functional", reference))
        mean = float(np.mean(values))
        half_width = 4.302652729911275 * float(np.std(values, ddof=1)) / math.sqrt(3)
        result.append({"method": "functional", "reference": reference, "n_seeds": 3,
                       "relative_cost_reductions_by_seed": json.dumps(dict(zip(SEEDS, values))),
                       "mean_relative_cost_reduction": mean, "ci95_low": mean-half_width, "ci95_high": mean+half_width,
                       "criterion": "relative measured C reduction>=0.10 and paired final quality within1pp",
                       "all_final_quality_guards_passed": all(quality_ok),
                       "scope": "conditional df2 t-interval; one shared rank search; no automatic hypothesis verdict"})
    return result


def admission():
    operators = operator_admission()
    spectra = {"shape": [128, 128, 3, 3], "norm_squared": 128.,
               "modes": [{"tail_squared": [float(128-r) for r in range(129)]} for _ in range(2)]}
    ranks, _ = energy_ranks(spectra, .10)
    assert ranks == (128, 128)  # No toy-grid cap96, even at an infeasible compressed grid.
    assert len(rule_candidates("energy", [spectra, spectra])) == 6
    batch = constant_batch_indices(2000, 32, 123)
    assert batch.shape == (32, 128)
    assert torch.equal(batch, constant_batch_indices(2000, 32, 123))
    assert storage_count(FULL_PAIR) > 128*128*9 + 256*256*9
    return {"H04_graph_output_input_factor_gradients": operators,
            "real_grid_and_spectral_rule": "passed", "effective_batch128_stream": "passed",
            "full_reserve_has_no_storage_reduction": "passed", "gpu_calls": False}


def snapshot_sources(output):
    paths = [Path(__file__), HERE / "H12_functional_rank_rgn.md", HERE / "run_h04_real.py",
             HERE / "run_local.py", HERE / "run_h06_real_feasibility.py",
             REPO / "tdecomp" / "tensor" / "tucker.py", REPO / "tdecomp" / "_base.py",
             REPO / "tdecomp" / "_random.py", REPO / "tests" / "test_h12_real_runner.py"]
    hashes = {}
    for source in paths:
        relative = source.relative_to(REPO)
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = file_hash(source)
    return hashes


def load_train_data(args, baseline):
    if file_hash(args.data / "cifar-10-python.tar.gz") != baseline["data_archive_sha256"]:
        raise ValueError("CIFAR archive hash differs from original baseline")
    if file_hash(args.baseline / "splits.json") != baseline["split_sha256"]:
        raise ValueError("original split file hash differs from baseline")
    split = json.loads((args.baseline / "splits.json").read_text(encoding="utf-8"))
    required = {"baseline": 40000, "calibration": 2000, "tuning": 3000, "recovery": 5000}
    # Existing baseline split naming is train rather than baseline.
    train_key = "train" if "train" in split else "baseline"
    actual_required = {train_key: 40000, **{k: v for k, v in required.items() if k != "baseline"}}
    joined = []
    for name, count in actual_required.items():
        if len(split[name]) != count:
            raise ValueError("original split size changed")
        joined.extend(split[name])
    if len(set(joined)) != 50000 or min(joined) != 0 or max(joined) != 49999:
        raise ValueError("original training split indices overlap or are out of range")
    norm = baseline["normalization"]
    normalize = transforms.Normalize(norm["mean"], norm["std"])
    plain_transform = transforms.Compose([transforms.ToTensor(), normalize])
    plain = datasets.CIFAR10(args.data, train=True, download=False, transform=plain_transform)
    augmented = datasets.CIFAR10(args.data, train=True, download=False, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    tune = DataLoader(Subset(plain, split["tuning"]), batch_size=128, shuffle=False, num_workers=0)
    provenance = {}
    for name in ("calibration", "tuning", "recovery"):
        indices = torch.as_tensor(split[name], dtype=torch.int64)
        provenance[name] = {"original_indices_sha256": tensor_hash(indices), "n": len(indices),
                            "original_uint8_images_sha256": hashlib.sha256(plain.data[indices.numpy()].tobytes()).hexdigest(),
                            "original_labels_sha256": tensor_hash(torch.as_tensor(plain.targets, dtype=torch.int64)[indices])}
    return plain, augmented, tune, split, plain_transform, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--microbatch", type=int, choices=(128, 64, 32), default=128)
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not args.admission_only and (args.baseline is None or args.data is None):
        parser.error("--baseline and --data are required for pilot/main")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    manifest = {"state": "admission", "hypothesis": "H12", "setting": "real", "device": args.device,
                "git_sha": git("rev-parse", "HEAD"), "git_status": git("status", "--short"), "base_sha": BASE_SHA,
                "source_hashes": snapshot_sources(args.output), "command": sys.argv,
                "torch": torch.__version__, "python": sys.version,
                "package_versions": {name: metadata.version(name) for name in ("torchvision", "tensorly", "numpy", "psutil")},
                "cuda_build_version": torch.version.cuda, "seeds": [101] if args.pilot else list(SEEDS),
                "selection_seed": 101, "rules": list(RULES), "functional_pairs": FUNCTIONAL_PAIRS,
                "fractions": FRACTIONS, "energy_epsilons": EPSILONS, "spatial_ranks": [3, 3],
                "candidate_steps": 32, "selected_calibration_steps": 128, "selected_recovery_steps": 400,
                "initial32_on_every_selected_searched_candidate": True,
                "full_rank_reserve_initial_steps_exception": 0, "microbatch": args.microbatch, "effective_batch": 128,
                "amortization_requests": HORIZON, "quality_accuracy_drop": QUALITY_DROP,
                "BN_policy": "original baseline statistics copied to every model; all BN eval; no recalibration",
                "final_test_opened": False,
                "limitations": ["selection uses528-update training cost forecast from32; exact optimum C is not claimed",
                                "rank search occurs once on seed101; shared search cost is charged in full to each seed's10000-request C",
                                "selected seed101 retains candidate32; fixed pair seeds202/303 receive32 initial calibration, not new search",
                                "functional/fractional/energy primary only; nuclear ADMM, RGN solver comparison and5-bootstrap stability are separate pending secondary stages",
                                "forced full-rank reserve receives no initial32 at any seed, only128/400; no seventh candidate training; forced reserve forecast has528 updates with seconds unknown until measured",
                                "mathematical parameter storage is reported separately from observed execution memory",
                                "torchvision actual version differs from protocol interface0.20"]}
    write_json(args.output / "manifest.json", manifest)
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=REPO, check=True)
        manifest["corrected_base_gate"] = "passed"
        write_json(args.output / "admission.json", admission())
        if args.baseline is not None:
            base_admission = baseline_admission(args.baseline)
            write_json(args.output / "baseline_admission.json", base_admission)
        if args.admission_only:
            if args.baseline is not None:
                original = torch.load(args.baseline / "seed-101" / "model.pt", map_location="cpu", weights_only=True)
                spectra = [original_channel_spectra(original[target + ".weight"].detach()) for target in TARGETS]
                write_json(args.output / "energy-original-spectra.json", spectra)
                write_json(args.output / "real_rank_plan_admission.json", {"checkpoint_sha256": base_admission["seeds"]["101"]["checkpoint_sha256"],
                           "rules": {rule: rule_candidates(rule, spectra if rule == "energy" else None) for rule in RULES},
                           "scope": "CPU weight-derived candidate plan only; no calibration/search/forward/data access", "final_test_opened": False})
            manifest["state"] = "admission_passed"
            return
        baseline = json.loads((args.baseline / "manifest.json").read_text(encoding="utf-8"))
        plain, augmented, tune, split, plain_transform, input_hashes = load_train_data(args, baseline)
        write_json(args.output / "input_provenance.json", input_hashes)
        manifest.update(archive_sha256=baseline["data_archive_sha256"], split_sha256=baseline["split_sha256"],
                        normalization=baseline["normalization"], baseline_admission_sha256=file_hash(args.output / "baseline_admission.json"))
        teacher = models.resnet18(weights=None, num_classes=10).to(args.device)
        teacher.load_state_dict(torch.load(args.baseline / "seed-101" / "model.pt", map_location=args.device, weights_only=True))
        teacher.eval()
        original101 = evaluate(teacher, tune, args.device)
        if original101["accuracy"] < .70:
            raise ArithmeticError("original seed101 teacher failed70% tuning gate")
        write_json(args.output / "original-seed-101-tuning.json", original101)
        timing_input = next(iter(tune))[0][:64].to(args.device)
        manifest["state"] = "running"
        write_json(args.output / "manifest.json", manifest)
        if args.pilot:
            guard = ResourceGuard(args.device)
            student, residuals = make_candidate(teacher, FUNCTIONAL_PAIRS[4], 101)
            phase, _ = train_updates(student, plain, split["calibration"], 50101, args.device, 20, guard, args.microbatch)
            torch.save(student.state_dict(), args.output / "pilot.pt")
            write_json(args.output / "pilot.json", {"pair": FUNCTIONAL_PAIRS[4], "residuals": residuals,
                       "training": phase, "tuning": evaluate(student, tune, args.device),
                       "checkpoint_sha256": file_hash(args.output / "pilot.pt"), **guard.metrics(), "final_test_opened": False})
            manifest["state"] = "pilot_completed"
            return
        selected = {}
        search = []
        confirmations = []
        for rule in RULES:
            guard = ResourceGuard(args.device, max_seconds=2700)
            search_started = time.perf_counter()
            spectra = None
            if rule == "energy":
                spectra = [original_channel_spectra(layer_at(teacher, target).weight.detach(), guard) for target in TARGETS]
                write_json(args.output / "energy-original-spectra.json", spectra)
            candidates = []
            for index, (pair, rule_evidence) in enumerate(rule_candidates(rule, spectra)):
                guard.check()
                synchronize(args.device)
                setup_started = time.perf_counter()
                student, residuals = make_candidate(teacher, pair, 101)
                synchronize(args.device)
                factor_seconds = time.perf_counter() - setup_started
                phase, _ = train_updates(student, plain, split["calibration"], 50101, args.device, 32, guard, args.microbatch)
                quality = evaluate(student, tune, args.device)
                timing = measured_latency(student, timing_input[:1], args.device, guard)
                checkpoint = args.output / f"search-{rule}-{index}.pt"
                torch.save(student.state_dict(), checkpoint)
                history_path = args.output / f"search-{rule}-{index}-history.json"
                write_json(history_path, phase)
                candidate = {"rule": rule, "index": index, "pair": pair, "rank_evidence": rule_evidence,
                             "calibration_steps": 32, "quality": quality,
                             "admissible": quality["accuracy"] >= original101["accuracy"] - QUALITY_DROP,
                             "factor_seconds": factor_seconds, "calibration_seconds": phase["seconds"],
                             "forecast_remaining528_seconds": phase["seconds"] * 528 / 32,
                             "forecast_cost_without_common_search": phase["seconds"] * 528 / 32 / HORIZON + timing["p50_ms"] / 1000,
                             "latency_batch1": timing, "factor_parameters": storage_count(pair),
                             "weight_residuals": residuals, "candidate_checkpoint": checkpoint.name,
                             "candidate_checkpoint_sha256": file_hash(checkpoint), **guard.metrics()}
                candidates.append(candidate)
                search.append(candidate)
                write_json(args.output / "search.json", search)
                del student
            chosen = choose_index(candidates)
            reserve = chosen is None
            if reserve:
                # No seventh32-step search: construct and validate untrained
                # exact reserve, then freeze its selection before128/400.
                synchronize(args.device)
                setup_started = time.perf_counter()
                student, residuals = make_candidate(teacher, FULL_PAIR, 101)
                synchronize(args.device)
                reserve_factor_seconds = time.perf_counter() - setup_started
                reserve_quality = evaluate(student, tune, args.device)
                if reserve_quality["accuracy"] < original101["accuracy"] - QUALITY_DROP:
                    raise ArithmeticError("untrained exact full-rank reserve failed original tuning quality")
                checkpoint = args.output / f"search-{rule}-reserve-untrained.pt"
                torch.save(student.state_dict(), checkpoint)
                selection = {"rule": rule, "index": "reserve", "pair": FULL_PAIR, "calibration_steps": 0,
                             "factor_seconds": reserve_factor_seconds, "quality": reserve_quality,
                             "candidate_checkpoint": checkpoint.name, "candidate_checkpoint_sha256": file_hash(checkpoint),
                             "weight_residuals": residuals, "forecast_remaining_updates": 528,
                             "forecast_remaining528_seconds": None,
                             "scope": "untrained full-rank reserve outside6-candidate search; no32-step reserve speed probe"}
                search.append(selection)
                write_json(args.output / "search.json", search)
                del student
            else:
                selection = candidates[chosen]
            synchronize(args.device)
            shared_search_seconds = time.perf_counter() - search_started
            guard.check()
            selected[rule] = {"selected": selection, "pair": selection["pair"], "full_rank_reserve": reserve,
                              "search_seconds": shared_search_seconds, "selection_seed": 101,
                              "frozen_before_confirming_seeds_and_test": True,
                              "objective": "minimum forecast C among original-tuning-accuracy minus1pp; full shared search price common to candidates"}
            write_json(args.output / "selection.json", selected)
        del teacher
        # All three rules and pairs have now been frozen using seed101 only.
        for seed in SEEDS:
            seed_all(seed, args.device)
            teacher = models.resnet18(weights=None, num_classes=10).to(args.device)
            teacher.load_state_dict(torch.load(args.baseline / f"seed-{seed}" / "model.pt", map_location=args.device, weights_only=True))
            teacher.eval()
            original = original101 if seed == 101 else evaluate(teacher, tune, args.device)
            if original["accuracy"] < .70:
                raise ArithmeticError("original teacher failed70% tuning gate")
            write_json(args.output / f"original-seed-{seed}-tuning.json", original)
            directory = args.output / f"seed-{seed}"
            directory.mkdir()
            for rule in RULES:
                guard = ResourceGuard(args.device)
                selection = selected[rule]
                pair = tuple(tuple(r) for r in selection["pair"])
                synchronize(args.device)
                factor_started = time.perf_counter()
                initial32 = None
                # For101 reuse the saved32-step candidate instead of training
                # a seventh/new candidate. Constructing shapes here is paid.
                if seed == 101:
                    candidate_path = args.output / selection["selected"]["candidate_checkpoint"]
                    if file_hash(candidate_path) != selection["selected"]["candidate_checkpoint_sha256"]:
                        raise ArithmeticError("selected candidate checkpoint changed after freeze")
                    student = restore_candidate(teacher, torch.load(candidate_path, map_location=args.device, weights_only=True))
                else:
                    student, residuals = make_candidate(teacher, pair, seed)
                synchronize(args.device)
                fixed_factor_seconds = time.perf_counter() - factor_started
                if seed != 101 and initial_steps(selection):
                    initial32, _ = train_updates(student, plain, split["calibration"], seed + 50000,
                                                  args.device, 32, guard, args.microbatch)
                before = evaluate(student, tune, args.device)
                # Fresh selected AdamW policy shared by all three rules. The
                # same optimizer continues through selected128 and recovery400.
                calibration, optimizer = train_updates(student, plain, split["calibration"], seed + 51000,
                                                        args.device, 128, guard, args.microbatch)
                recovery, _ = train_updates(student, augmented, split["recovery"], seed + 60000,
                                             args.device, 400, guard, args.microbatch, optimizer)
                tuning_after = evaluate(student, tune, args.device)
                timing1 = measured_latency(student, timing_input[:1], args.device, guard)
                timing64 = measured_latency(student, timing_input, args.device, guard)
                checkpoint = directory / f"{rule}.pt"
                torch.save(student.state_dict(), checkpoint)
                checkpoint_sha = file_hash(checkpoint)
                write_json(directory / f"{rule}-history.json", {"initial32": initial32, "selected_calibration": calibration,
                                                               "recovery": recovery, "initial32_reused_from_search": seed == 101 and not selection["full_rank_reserve"]})
                total_extra = (selection["search_seconds"] + fixed_factor_seconds +
                               (initial32["seconds"] if initial32 else 0) + calibration["seconds"] + recovery["seconds"])
                row = {"seed": seed, "rule": rule, "pair": pair, "full_rank_reserve": selection["full_rank_reserve"],
                       "selection_seed": 101, "new_rank_search": False, "original_tuning": original,
                       "tuning_before_selected128": before, "tuning_after": tuning_after,
                       "quality_passed": tuning_after["accuracy"] >= original["accuracy"] - QUALITY_DROP,
                       "shared_search_seconds_charged": selection["search_seconds"], "fixed_factor_seconds": fixed_factor_seconds,
                       "fixed_initial32_seconds": initial32["seconds"] if initial32 else 0,
                       "initial_calibration_updates": initial_steps(selection), "selected_calibration_updates": 128, "recovery_updates": 400,
                       "selected_calibration_seconds": calibration["seconds"], "recovery_seconds": recovery["seconds"],
                       "measured_additional_training_and_search_seconds": total_extra,
                       "measured_cost_seconds_per_image": total_extra / HORIZON + timing1["p50_ms"] / 1000,
                       "amortization_requests": HORIZON, "latency_batch1": timing1, "latency_batch64_secondary": timing64,
                       "factor_parameters": storage_count(pair), "dense_target_coefficients": 128*128*9 + 256*256*9,
                       "mathematical_storage_ratio": storage_count(pair) / (128*128*9 + 256*256*9),
                       "mathematical_storage_reduction": storage_count(pair) < (128*128*9 + 256*256*9),
                       "checkpoint_sha256": checkpoint_sha, "checkpoint": str(checkpoint.relative_to(args.output)),
                       "status": "completed", **guard.metrics()}
                confirmations.append(row)
                write_json(args.output / "runs.json", confirmations)
                print(json.dumps(row), flush=True)
                del student, optimizer
            del teacher
        # Persistence/hash and shape locks for every confirmed checkpoint,
        # before opening official test data even once.
        for row in confirmations:
            if file_hash(args.output / row["checkpoint"]) != row["checkpoint_sha256"]:
                raise ArithmeticError("confirmed checkpoint changed before final test")
        manifest["all_confirmed_checkpoints_frozen_before_test"] = True
        manifest["final_test_opened"] = True
        write_json(args.output / "manifest.json", manifest)
        official = datasets.CIFAR10(args.data, train=False, download=False, transform=plain_transform)
        test = DataLoader(official, batch_size=128, shuffle=False, num_workers=0)
        final = []
        for seed in SEEDS:
            guard = ResourceGuard(args.device)
            teacher = models.resnet18(weights=None, num_classes=10).to(args.device)
            teacher.load_state_dict(torch.load(args.baseline / f"seed-{seed}" / "model.pt", map_location=args.device, weights_only=True))
            teacher.eval()
            teacher_quality = evaluate(teacher, test, args.device)
            original_row = {"seed": seed, "rule": "original", "final_test": teacher_quality,
                            "baseline_checkpoint_sha256": base_admission["seeds"][str(seed)]["checkpoint_sha256"]}
            final.append(original_row)
            for rule in RULES:
                row = next(item for item in confirmations if item["seed"] == seed and item["rule"] == rule)
                student = restore_candidate(teacher, torch.load(args.output / row["checkpoint"], map_location=args.device, weights_only=True))
                quality = evaluate(student, test, args.device)
                final.append({"seed": seed, "rule": rule, "pair": row["pair"], "final_test": quality,
                              "quality_passed": quality["accuracy"] >= teacher_quality["accuracy"] - QUALITY_DROP,
                              "checkpoint_sha256": row["checkpoint_sha256"], "selection_and_checkpoint_frozen": True})
                write_json(args.output / "final.json", final)
                guard.check()
                del student
            del teacher
        # Per-layer CSV records keep actual primary search provenance explicit.
        table = []
        for item in search:
            for target, ranks in zip(TARGETS, item["pair"]):
                table.append({"seed": 101, "policy": item["rule"], "candidate_id": item["index"], "layer": target,
                              "rank_tuple": json.dumps([*ranks, 3, 3]), "criterion": "tuning accuracy and forecast C",
                              "calibration_steps": item["calibration_steps"], "tuning_loss": item["quality"]["cross_entropy"],
                              "tuning_accuracy": item["quality"]["accuracy"], "param_count_two_layers": storage_count(item["pair"]),
                              "factor_seconds": item["factor_seconds"], "calibration_seconds": item.get("calibration_seconds", 0),
                              "search_seconds_shared": selected[item["rule"]]["search_seconds"], "amortization_requests": HORIZON,
                              "forecast_cost_without_common_search": item.get("forecast_cost_without_common_search"),
                              "selected": item["index"] == selected[item["rule"]]["selected"]["index"]})
        write_csv(args.output / "rank_search.csv", table)
        write_csv(args.output / "paired_summary.csv", paired_cost_summary(confirmations, final))
        pareto = []
        for row in confirmations:
            same_seed = [other for other in confirmations if other["seed"] == row["seed"]]
            loss = row["original_tuning"]["accuracy"] - row["tuning_after"]["accuracy"]
            dominated = any(other["tuning_after"]["accuracy"] >= row["tuning_after"]["accuracy"]
                            and other["factor_parameters"] <= row["factor_parameters"]
                            and other["measured_cost_seconds_per_image"] <= row["measured_cost_seconds_per_image"]
                            and (other["tuning_after"]["accuracy"] > row["tuning_after"]["accuracy"]
                                 or other["factor_parameters"] < row["factor_parameters"]
                                 or other["measured_cost_seconds_per_image"] < row["measured_cost_seconds_per_image"])
                            for other in same_seed if other["rule"] != row["rule"])
            pareto.append({"seed": row["seed"], "rule": row["rule"], "quality_loss_accuracy_fraction": loss,
                           "mathematical_factor_fp32_bytes": 4*row["factor_parameters"],
                           "actual_peak_gpu_reserved_bytes": row["peak_gpu_reserved_bytes"],
                           "total_cost_seconds_per_image": row["measured_cost_seconds_per_image"],
                           "dominated": dominated, "split": "tuning", "scope": "secondary; quality/count/C among3 frozen rules"})
        write_csv(args.output / "pareto.csv", pareto)
        write_json(args.output / "table_units.json", {"time_fields": "seconds except latency *_ms", "accuracy": "fraction", "cost": "seconds per single image over10000 requests", "coefficients": "FP32 mathematical factor/core count"})
        manifest["state"] = "completed"
    except Exception as error:
        manifest.update(state="resource_failure" if isinstance(error, (MemoryError, TimeoutError)) else "implementation_failure",
                        error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
