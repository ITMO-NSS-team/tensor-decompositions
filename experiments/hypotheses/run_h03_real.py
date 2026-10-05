"""H03 pinned GPT2/WikiText2, global mean-preserving rotation and block5 int4 QAT.

All residual branches, embeddings, learned positions and head share one Q.
Quantization uses dequantized FP32 weights; hardware int4 speed is not measured.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow.parquet as pq
import psutil
import torch
from torch.nn import functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from experiments.hypotheses.gpt2_rotation_core import (
    CONVOLUTIONS, CalibrationBlock, cache_block_inputs, choose_rotation, complement,
    disable_dropout, global_rotation_checks, install_quantization, logits,
    rotate_global_in_place, small_admission, untie_and_absorb,
)
from experiments.hypotheses.run_h02_synthetic import (
    BASE_SHA, ResourceGuard, file_hash, git, synchronize, tensor_hash, write_csv, write_json,
)
from experiments.hypotheses.run_h03_synthetic import METHODS, quant4

MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MODEL_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
SEEDS = (101, 202, 303)
DOCUMENT_TITLE = re.compile(r"^\s*=\s+[^=\s].*?\s+=\s*$")


def documents_from_rows(rows):
    """Keep every raw row and split only level-one WikiText title boundaries."""
    documents, current = [], []
    for row in rows:
        if DOCUMENT_TITLE.match(row) and any(value.strip() for value in current):
            documents.append("".join(current))
            current = []
        current.append(row)
    if current:
        documents.append("".join(current))
    assert "".join(documents) == "".join(rows)
    return documents


def tokenize_rows(rows, tokenizer):
    documents = documents_from_rows(rows)
    tokens, boundaries = [], []
    for index, document in enumerate(documents):
        tokens.extend(tokenizer.encode(document, add_special_tokens=False, truncation=False))
        if index + 1 < len(documents):
            tokens.append(tokenizer.eos_token_id)
        boundaries.append(len(tokens))
    return torch.tensor(tokens, dtype=torch.long), boundaries


def window_stream(tokens, length=128):
    complete = tokens.numel() // length
    return tokens[:complete * length].reshape(complete, length).contiguous(), tokens.numel() % length


def load_windows(data_root, out, *, include_test=True):
    provenance_path = data_root / "gpt2-wikitext-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["gpt2_revision"] != MODEL_REVISION or provenance["wikitext_revision"] != DATA_REVISION:
        raise ValueError("downloaded data/model revisions differ from protocol")
    for record in provenance["files"]:
        path = Path(record["path"])
        if not path.is_file() or file_hash(path) != record["sha256"]:
            raise ValueError(f"download provenance hash mismatch: {path.name}")
    if file_hash(data_root / "gpt2" / "model.safetensors") != MODEL_SHA:
        raise ValueError("GPT2 checkpoint does not match pinned download")
    tokenizer = GPT2TokenizerFast.from_pretrained(data_root / "gpt2", local_files_only=True)
    tokenizer.model_max_length = 10**9  # Entire documents are tokenized without truncation.
    streams, metadata = {}, {"provenance_sha256": file_hash(provenance_path), "splits": {}}
    for name in (("train", "validation", "test") if include_test else ("train", "validation")):
        path = data_root / "wikitext2" / "wikitext-2-raw-v1" / f"{name}-00000-of-00001.parquet"
        rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        tokens, boundaries = tokenize_rows(rows, tokenizer)
        windows, tail = window_stream(tokens)
        streams[name] = windows
        metadata["splits"][name] = {"raw_rows": len(rows), "blank_rows_preserved": sum(not row.strip() for row in rows),
                "documents": len(boundaries), "eos_insertions": max(0, len(boundaries) - 1), "document_end_offsets": boundaries,
                "source_sha256": file_hash(path), "tokens": tokens.numel(), "tokens_sha256": tensor_hash(tokens),
                "complete_windows": len(windows), "excluded_incomplete_tail_tokens": tail,
                "raw_strings_preserved": True, "document_rule": "split before level-one '= title =' rows; insert eos between documents"}
        torch.save(tokens, out / f"{name}-tokens.pt")
    if len(streams["train"]) < 1152:
        raise ValueError("fewer than 147456 training tokens; protocol sample cannot be shortened")
    result = {"recovery": streams["train"][:1024], "calibration": streams["train"][1024:1152],
              "tuning": streams["validation"]}
    if include_test:
        result["test"] = streams["test"]
    metadata["training_partition"] = {"recovery_tokens": 131072, "calibration_tokens": 16384,
             "unused_complete_training_tokens": max(0, len(streams["train"]) - 1152) * 128}
    metadata["window_prediction_policy"] = "128-token disjoint windows; causal CE predicts positions1..127 from0..126"
    metadata["inputs"] = {name: {"windows_sha256": tensor_hash(value), "windows": len(value),
                                      "input_tokens": value.numel(), "prediction_tokens": len(value) * 127}
                          for name, value in result.items()}
    write_json(out / "inputs.json", metadata)
    return result, metadata


def batches(seed, count, steps, batch_size=8):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(count, (steps, batch_size), generator=gen)


def phase_steps(args):
    if args.recovery_pilot:
        return 0, 20
    return (20, 0) if args.pilot else (50, 200)


def autocast(device, enabled):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                          enabled=enabled and torch.device(device).type == "cuda")


def check_gradients(parameters):
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ArithmeticError("nonfinite parameter gradient")


@torch.no_grad()
def quality(model, windows, device, *, batch_size=8, teacher=None, guard=None):
    previous = model.training
    model.eval()
    total = error = energy = 0.0
    count = 0
    try:
        for start in range(0, len(windows), batch_size):
            if guard:
                guard.check()
            tokens = windows[start:start + batch_size].to(device)
            prediction = logits(model, tokens).float()
            if not torch.isfinite(prediction).all():
                raise ArithmeticError("nonfinite GPT2 logits")
            total += float(F.cross_entropy(prediction[:, :-1].reshape(-1, prediction.shape[-1]),
                                         tokens[:, 1:].reshape(-1), reduction="sum"))
            count += tokens.shape[0] * (tokens.shape[1] - 1)
            if teacher is not None:
                exact = logits(teacher, tokens).float()
                error += float((prediction - exact).square().sum())
                energy += float(exact.square().sum())
    finally:
        model.train(previous)
    if not count:
        raise ValueError("quality split has no complete prediction windows")
    ce = total / count
    return {"cross_entropy": ce, "perplexity": math.exp(ce), "windows": len(windows),
            "prediction_tokens": count, "logits_normalized_mse": error / energy if teacher is not None and energy else None}


def calibrate(block, hidden, target, seed, device, *, steps=50, microbatch=8, guard=None):
    block.train()
    parameters = list(block.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0)
    indices = batches(seed + 50000, len(hidden), steps)
    losses = []
    for ix in indices:
        if guard:
            guard.check()
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for start in range(0, len(ix), microbatch):
            selected = ix[start:start + microbatch]
            # Rotation/QR/SVD/calibration loss run FP32, matching the protocol.
            predicted = block(hidden[selected].to(device))
            loss = F.mse_loss(predicted.float(), target[selected].to(device)) * len(selected) / len(ix)
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite block calibration loss")
            loss.backward()
            value += float(loss.detach())
        check_gradients(parameters)
        optimizer.step()
        losses.append(value)
    return {"steps": steps, "batches_sha256": tensor_hash(indices), "losses": losses,
            "effective_batch": 8, "microbatch": microbatch,
            "block_parameters": sum(parameter.numel() for parameter in block.block.parameters()),
            "extra_cayley_coordinates": block.skew_coordinates.numel() if block.cayley else 0}


def freeze_checkpoint(model, path):
    # GPU optimizer/gradients never copied into a second live GPU model.
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)
    return file_hash(path)


def recover(model, windows, tuning, threshold, seed, device, checkpoint, *, steps=200, microbatch=8,
            bf16=False, guard=None):
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=.00005, weight_decay=.01)
    indices = batches(seed + 60000, len(windows), steps)
    synchronize(device)
    start_time = time.perf_counter()
    initial = quality(model, tuning, device, batch_size=microbatch, guard=guard)
    history = [{"step": 0, **initial}]
    hit = {"step": 0, "seconds": time.perf_counter() - start_time} if initial["perplexity"] <= threshold else None
    checkpoint_hash = freeze_checkpoint(model, checkpoint) if hit else None
    if hit:
        hit["seconds"] = time.perf_counter() - start_time
    for step, ix in enumerate(indices, 1):
        if guard:
            guard.check()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for offset in range(0, len(ix), microbatch):
            selected = ix[offset:offset + microbatch]
            tokens = windows[selected].to(device)
            with autocast(device, bf16):
                predicted = logits(model, tokens)
                loss = F.cross_entropy(predicted[:, :-1].float().reshape(-1, predicted.shape[-1]),
                                       tokens[:, 1:].reshape(-1)) * len(selected) / len(ix)
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite recovery loss")
            loss.backward()
            loss_value += float(loss.detach())
        check_gradients(parameters)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            result = quality(model, tuning, device, batch_size=microbatch, guard=guard)
            history.append({"step": step, "train_cross_entropy": loss_value, **result})
            if hit is None and result["perplexity"] <= threshold:
                synchronize(device)
                hit = {"step": step, "seconds": time.perf_counter() - start_time}
                checkpoint_hash = freeze_checkpoint(model, checkpoint)
                hit["seconds"] = time.perf_counter() - start_time
    if hit is None:
        checkpoint_hash = freeze_checkpoint(model, checkpoint)
    synchronize(device)
    seconds = time.perf_counter() - start_time
    del optimizer
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return history, hit, seconds, checkpoint_hash, tensor_hash(indices)


@torch.no_grad()
def packed_weights(model):
    result = {}
    for parent, name, _ in CONVOLUTIONS:
        weight = getattr(getattr(model.transformer.h[5], parent), name).weight.detach().cpu()
        scale = weight.abs().amax(0) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        integers = (weight / safe[None, :]).round().clamp(-7, 7).to(torch.int8)
        unsigned = (integers.to(torch.int16) + 8).to(torch.uint8).reshape(-1)
        if unsigned.numel() % 2:
            unsigned = F.pad(unsigned, (0, 1))
        packed = unsigned[::2] | (unsigned[1::2] << 4)
        decoded = torch.stack((packed & 15, packed >> 4), 1).reshape(-1)[:weight.numel()].to(torch.int16) - 8
        restored = decoded.reshape(weight.shape).float() * scale[None, :]
        torch.testing.assert_close(restored, quant4(weight, channel_axis=1), atol=0, rtol=0)
        result[f"{parent}.{name}"] = {"packed_uint8": packed, "scales": scale, "shape": tuple(weight.shape),
                                     "channel_axis": 1, "signed_offset": 8}
    return result


@torch.no_grad()
def latency(model, tokens, device, guard=None):
    values = []
    for step in range(230):
        if guard:
            guard.check()
        synchronize(device)
        start = time.perf_counter()
        logits(model, tokens)
        synchronize(device)
        if step >= 30:
            values.append((time.perf_counter() - start) * 1000)
    measured = torch.tensor(values)
    return {"p50_ms": float(measured.quantile(.5)), "p95_ms": float(measured.quantile(.95)),
            "warmups": 30, "synchronized_repeats": 200, "batch_size": len(tokens), "context": tokens.shape[1]}


def paired_summary(rows, expected_seeds):
    summary = []
    for method in ("hadamard", "procrustes", "cayley"):
        for reference in ("identity", "haar"):
            pairs = []
            for seed in expected_seeds:
                left = next((r for r in rows if r["seed"] == seed and r["method"] == method), {})
                right = next((r for r in rows if r["seed"] == seed and r["method"] == reference), {})
                if left.get("quality_reached") and right.get("quality_reached"):
                    pairs.append((right["total_seconds_to_quality"] - left["total_seconds_to_quality"]) / right["total_seconds_to_quality"])
            complete = len(pairs) == 3 and len(expected_seeds) == 3
            mean = sum(pairs) / len(pairs) if complete else None
            half = 4.302652729911275 * math.sqrt(sum((v - mean)**2 for v in pairs) / 2) / math.sqrt(3) if complete else None
            summary.append({"method": method, "reference": reference, "n_seeds": len(pairs),
                            "paired_relative_cost_reductions": json.dumps(pairs), "mean_delta": mean,
                            "ci95_low": mean - half if complete else None, "ci95_high": mean + half if complete else None,
                            "criterion": "relative cost reduction >=0.10 with every pair reaching tuning quality",
                            "outcome": "indeterminate" if not complete else ("cost_criterion_passed" if mean - half >= .10 else "cost_criterion_not_established")})
    return summary


def run_method(teacher, data, hidden, targets, dense_tuning, seed, method, args, shared_seconds):
    directory = args.out / f"seed-{seed}" / method
    directory.mkdir(parents=True)
    guard = ResourceGuard(args.device, max_seconds=2400)
    row = {"hypothesis": "H03", "stage": "real", "seed": seed, "method": method,
           "status": "running", "stop_reason": "", "quant_bits": 4,
           "norm_type": "LayerNorm_affine_absorbed", "position_type": "learned_positions",
           "compensation": "global_all12blocks_embeddings_positions_head", "data_split": "tuning"}
    base = block = None
    calibration_steps, recovery_steps = phase_steps(args)
    try:
        synchronize(args.device)
        start = time.perf_counter()
        base = untie_and_absorb(teacher)
        e = complement(base.config.n_embd, hidden)
        assembly_dtype = torch.float64 if args.rotation_assembly_fp64 else torch.float32
        q = choose_rotation(method, hidden.reshape(-1, base.config.n_embd).to(args.device), seed, e.to(args.device),
                            assembly_dtype=assembly_dtype)
        row.update(global_rotation_checks(base, q, data["calibration"][:args.microbatch].to(args.device), original=teacher))
        block = CalibrationBlock(base.transformer.h[5], q, cayley=method == "cayley",
                                 assembly_dtype=assembly_dtype).to(args.device)
        synchronize(args.device)
        row["setup_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        calibration = calibrate(block, hidden, targets, seed, args.device, steps=calibration_steps,
                                microbatch=args.microbatch, guard=guard)
        synchronize(args.device)
        row["calibration_seconds"] = time.perf_counter() - start
        write_json(directory / "calibration_history.json", calibration)
        materialization_start = time.perf_counter()
        block.commit_unrotated_weights(base.transformer.h[5])
        q = block.rotation().detach()
        row["postcalibration_global_admission"] = global_rotation_checks(base, q, data["calibration"][:args.microbatch].to(args.device))
        row["extra_cayley_coordinates"] = calibration["extra_cayley_coordinates"]
        del block
        block = None
        rotate_global_in_place(base, q)
        install_quantization(base)
        base.eval()
        synchronize(args.device)
        row["postcalibration_admission_and_materialization_seconds"] = time.perf_counter() - materialization_start
        row["parameters_after_affine_absorption"] = sum(parameter.numel() for parameter in base.parameters())
        if row["parameters_after_affine_absorption"] != 163049041:
            raise ValueError("affine-absorbed untied parameter count differs from derived count")
        checkpoint = directory / "checkpoint.pt"
        history, hit, seconds, checkpoint_hash, order_hash = recover(base, data["recovery"], data["tuning"],
            1.02 * dense_tuning["perplexity"], seed, args.device, checkpoint,
            steps=recovery_steps, microbatch=args.microbatch, bf16=args.bf16_recovery, guard=guard)
        row.update(recovery_seconds=seconds, recovery_batches_sha256=order_hash, checkpoint_sha256=checkpoint_hash,
                   quality_reached=hit is not None, recovery_step_to_quality=hit["step"] if hit else None,
                   calibration_steps=calibration_steps, recovery_steps=recovery_steps,
                   censored_at_steps=None if hit else recovery_steps,
                   shared_preparation_seconds=shared_seconds)
        # Shared data/cache/baseline preparation, verification and packing are paid.
        overhead_start = time.perf_counter()
        torch.save(q.cpu(), directory / "rotation.pt")
        packed = packed_weights(base)
        torch.save(packed, directory / "packed_int4.pt")
        row["raw_packed_weight_and_scale_bytes"] = sum(v["packed_uint8"].numel() + v["scales"].numel() * 4 for v in packed.values())
        row["packing_seconds"] = time.perf_counter() - overhead_start
        row["total_seconds_to_quality"] = (shared_seconds + row["setup_seconds"] + row["calibration_seconds"]
            + row["postcalibration_admission_and_materialization_seconds"] + row["packing_seconds"] + hit["seconds"]) if hit else None
        write_json(directory / "recovery_history.json", history)
        # Checkpoint and rotation are fixed and hashed BEFORE opening this final evaluation.
        row["rotation_sha256"] = file_hash(directory / "rotation.pt")
        row["packed_sha256"] = file_hash(directory / "packed_int4.pt")
        final_split = "tuning" if args.pilot else "test"
        final = quality(base, data[final_split], args.device, batch_size=args.microbatch, teacher=teacher, guard=guard)
        write_json(directory / "final_metrics.json", {"data_split": final_split, **final})
        row.update(task_loss=final["cross_entropy"], test_perplexity=final["perplexity"],
                   logits_normalized_mse=final["logits_normalized_mse"], data_split=final_split)
        if not args.pilot:
            row["latency_batch1"] = latency(base, data["tuning"][:1].to(args.device), args.device, guard)
            row["latency_batch8"] = latency(base, data["tuning"][:8].to(args.device), args.device, guard)
        row.update(guard.metrics(), status="pilot_complete" if args.pilot else "complete")
        row["total_seconds"] = time.perf_counter() - guard.start
    except (ArithmeticError, AssertionError, MemoryError, RuntimeError, TimeoutError, ValueError) as error:
        row.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
        write_json(directory / "failure.json", {"row": row, "traceback": traceback.format_exc()})
    finally:
        del base, block
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO.parent / "audit" / "datasets")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--real-admission", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--recovery-pilot", action="store_true",
                        help="identity/Haar, no finaltest, zero calibration and20full-model AdamW updates")
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--bf16-recovery", action="store_true")
    parser.add_argument("--rotation-assembly-fp64", action="store_true",
                        help="explicit FP64 assembly of smallQ then FP32 cast; QR/SVD/solve/weights/states stayFP32")
    parser.add_argument("--seeds", default="101,202,303")
    args = parser.parse_args(argv)
    if args.pilot and args.recovery_pilot:
        parser.error("choose either block-calibration pilot or full-model recovery pilot")
    args.pilot = args.pilot or args.recovery_pilot
    calibration_steps, recovery_steps = phase_steps(args)
    args.seeds = (101,) if args.pilot or args.real_admission else tuple(int(value) for value in args.seeds.split(","))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(SEEDS):
        parser.error("seeds must be distinct protocol seeds101/202/303")
    if transformers.__version__ != "4.39.1":
        raise RuntimeError("the reviewed runner requires transformers4.39.1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], check=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name("gpt2_rotation_core.py"),
             Path(__file__).with_name("run_h03_synthetic.py"), Path(__file__).with_name("run_h02_synthetic.py"),
             Path(__file__).with_name("H03_rotations_recovery.md")]
    for path in files:
        shutil.copy2(path, args.out / (path.name + ".source"))
    native_source = Path(inspect.getfile(GPT2LMHeadModel))
    shutil.copy2(native_source, args.out / "modeling_gpt2.py.source")
    driver_snapshot = None
    if args.device == "cuda":
        driver_snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                                    "--format=csv,noheader,nounits"], text=True).strip()
    manifest = {"hypothesis": "H03", "setting": "real", "status": "admission", "command": sys.argv,
            "git_sha": git("rev-parse", "HEAD"), "base_sha": BASE_SHA,
            "source_hashes": {p.name: file_hash(p) for p in files}, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "native_gpt2_source_sha256": file_hash(inspect.getfile(GPT2LMHeadModel)),
            "driver_snapshot": driver_snapshot, "ram_total_bytes": psutil.virtual_memory().total,
            "cpu_threads": torch.get_num_threads(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "device": args.device, "dtype": "FP32 parameters/gradients/Adam/QR/SVD",
            "bf16_recovery_autocast": args.bf16_recovery, "seeds": args.seeds, "effective_batch": 8,
            "rotation_assembly": "FP64 analytic Householder+mean projection thenFP32cast" if args.rotation_assembly_fp64 else "FP32 analytic Householder+mean projection",
            "rotation_assembly_uniform_all_methods": True,
            "microbatch": args.microbatch, "context": 128, "model_revision": MODEL_REVISION,
            "data_revision": DATA_REVISION, "calibration_steps": calibration_steps,
            "recovery_steps": recovery_steps, "threshold": "tuning PPL<=1.02*dense_tuning_PPL",
            "mode": "full_model_recovery_pilot" if args.recovery_pilot else ("block_calibration_pilot" if args.pilot else "confirm"),
            "final_test_enabled": not (args.pilot or args.real_admission or args.admission_only),
            "quality_observation_steps": list(range(0, recovery_steps + 1, 10)),
            "whole_hypothesis_outcome": "indeterminate",
            "limitations": ["one common pretrained checkpoint; three interventions, not independently trained GPT2 models",
                "global dense Q compensation includes all12blocks; internal attention heads and GELU coordinates unchanged",
                "only block5 four Conv1D matrices quantized; biases, embeddings and head remain FP32",
                "fake int4 training/dequantized execution; no native int4 hardware speed claim",
                "Cayley adds293761coordinates for50calibration steps; identity is fixed-A=0 ablation",
                "microbatch chosen before main; OOM stops a branch instead of silently changing a paired protocol",
                "tuning assessed at0/10/.../200updates; quality time resolved only on this grid",
                "raw packed bytes exclude serialization and other unquantized parameters",
                "RMSNorm/RoPE full neural control series remain separate",
                "n=3paired t interval is conditional on approximate normality; weak uncertainty estimate"]}
    write_json(args.out / "manifest.json", manifest)
    rows = []
    try:
        manifest["small_admission"] = small_admission()
        if args.admission_only:
            manifest["status"] = "admission_passed"
            return 0
        preparation_start = time.perf_counter()
        data, input_metadata = load_windows(args.data, args.out, include_test=not (args.pilot or args.real_admission))
        teacher = GPT2LMHeadModel.from_pretrained(args.data / "gpt2", local_files_only=True).to(args.device)
        disable_dropout(teacher)
        manifest["original_parameters"] = sum(parameter.numel() for parameter in teacher.parameters())
        manifest["untied_parameters_before_affine_absorption"] = 163037184
        if manifest["original_parameters"] != 124439808:
            raise ValueError("pinned GPT2 unique parameter count differs from protocol")
        if (teacher.config.n_embd, teacher.config.n_layer, teacher.config.n_head, teacher.config.vocab_size) != (768, 12, 12, 50257):
            raise ValueError("pinned GPT2 architecture differs from protocol")
        preparation_guard = ResourceGuard(args.device, max_seconds=2400)
        hidden, targets = cache_block_inputs(teacher, data["calibration"], args.device, batch_size=args.microbatch,
                                            guard=preparation_guard)
        torch.save({"hidden": hidden, "teacher_block_output": targets}, args.out / "calibration-cache.pt")
        manifest["calibration_cache"] = {"hidden_sha256": tensor_hash(hidden), "target_sha256": tensor_hash(targets),
                                         "file_sha256": file_hash(args.out / "calibration-cache.pt")}
        if args.real_admission:
            base = untie_and_absorb(teacher)
            e = complement(768, hidden).to(args.device)
            manifest["real_admission"] = {}
            for method in METHODS:
                try:
                    check = global_rotation_checks(base,
                        choose_rotation(method, hidden.reshape(-1, 768).to(args.device), 101, e,
                                        assembly_dtype=torch.float64 if args.rotation_assembly_fp64 else torch.float32),
                        data["calibration"][:args.microbatch].to(args.device), original=teacher)
                    manifest["real_admission"][method] = {"status": "passed", **check}
                except (ArithmeticError, AssertionError, RuntimeError, ValueError) as error:
                    manifest["real_admission"][method] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
                    write_json(args.out / "manifest.json", manifest)
                    raise
                write_json(args.out / "manifest.json", manifest)
            manifest["parameters_after_affine_absorption"] = sum(parameter.numel() for parameter in base.parameters())
            manifest["status"] = "real_admission_passed"
            return 0
        dense_tuning = quality(teacher, data["tuning"], args.device, batch_size=args.microbatch, guard=preparation_guard)
        write_json(args.out / "dense_tuning.json", dense_tuning)
        shared_seconds = time.perf_counter() - preparation_start
        manifest.update(status="running", shared_preparation_seconds=shared_seconds, inputs=input_metadata["inputs"])
        write_json(args.out / "manifest.json", manifest)
        for seed in args.seeds:
            for method in (("identity", "haar") if args.pilot else METHODS):
                row = run_method(teacher, data, hidden, targets, dense_tuning, seed, method, args, shared_seconds)
                rows.append(row)
                write_csv(args.out / "runs.csv", rows)
                print(json.dumps({"seed": seed, "method": method, "status": row["status"], "reason": row["stop_reason"]}), flush=True)
        if not args.pilot:
            dense_final = quality(teacher, data["test"], args.device, batch_size=args.microbatch)
            write_json(args.out / "dense_final_test.json", dense_final)
            for row in rows:
                if row.get("test_perplexity") is not None:
                    row["final_quality_passed"] = row["test_perplexity"] <= 1.02 * dense_final["perplexity"]
            write_csv(args.out / "paired_summary.csv", paired_summary(rows, args.seeds))
        write_csv(args.out / "runs.csv", rows)
        write_json(args.out / "units.json", {"seconds": "wall-clock seconds", "p50_ms_p95_ms": "milliseconds",
            "bytes": "bytes", "perplexity": "exp(mean natural-log next-token CE)",
            "mean_preservation_relative_error": "||Q1-1||2/sqrt(width)",
            "paired_delta": "(reference total quality cost - method total quality cost)/reference cost"})
        manifest["status"] = "complete" if all(row["status"] in ("complete", "pilot_complete") for row in rows) else "partial"
    except BaseException as error:
        manifest.update(status="interrupted", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
