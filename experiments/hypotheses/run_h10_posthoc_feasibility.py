"""Posthoc CPU exhaustive load-feasibility audit of frozen H10 Switch traces.

Each sparse layer assigns eight indivisible experts to four labeled devices,
exactly two per device. All2520 assignments are checked. Joint scenarios
require the SAME assignment to obey each split's OWN load threshold; split
loads are never pooled. Future traces are a retrospective oracle only.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import csv
import hashlib
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
SPLITS = ("calibration", "tuning", "future")
SCENARIOS = {"calibration": ("calibration",),
             "calibration_and_tuning": ("calibration", "tuning"),
             "calibration_and_tuning_and_future_posthoc_oracle": SPLITS}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_provenance(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "values_sha256": hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def balanced_placements(experts=8, devices=4, capacity=2):
    if experts != devices * capacity or min(experts, devices, capacity) <= 0:
        raise ValueError("exactly full equal expert slots are required")
    assignments = []
    def extend(remaining, destination, current):
        if destination == devices:
            assignments.append(tuple(current))
            return
        for group in itertools.combinations(remaining, capacity):
            updated = current.copy()
            for expert in group:
                updated[expert] = destination
            extend(tuple(expert for expert in remaining if expert not in group), destination+1, updated)
    extend(tuple(range(experts)), 0, [-1]*experts)
    assignments.sort()
    expected = math.factorial(experts) // math.factorial(capacity)**devices
    if len(assignments) != expected or len(set(assignments)) != expected:
        raise ArithmeticError("enumeration count or uniqueness invariant failed")
    return torch.tensor(assignments, dtype=torch.int64)


def integer_counts(tensor, expected_windows, expected_dispatches):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim != 4
            or tuple(tensor.shape[1:]) != (4,8,12) or tensor.shape[0] != expected_windows):
        raise ValueError("expected completed CPU counts[window,4,8,12]")
    if not torch.isfinite(tensor).all() or bool((tensor < 0).any()):
        raise ArithmeticError("accepted dispatch counts must be finite and nonnegative")
    integers = tensor.to(torch.int64)
    if not torch.equal(tensor, integers.to(tensor.dtype)):
        raise ArithmeticError("accepted dispatch counts must be integral")
    if int(integers.sum()) != expected_dispatches:
        raise ValueError("saved accepted-dispatch total does not match completed trace metadata")
    # One chunk has one origin; origins are not destinations and are summed
    # only to form expert loads, without changing accepted dispatches.
    if bool(((integers.sum((2,3)) > 0).sum(1) > 1).any()):
        raise ValueError("a single trace window has multiple original origin shards")
    return integers


def load_seed(source, seed):
    directory = source / f"seed-{seed}"
    names = ("statistics.pt", "trace-info.json", "future-statistics.pt", "future-info.json", "fixed-placements.json")
    missing = [name for name in names if not (directory/name).is_file()]
    if missing:
        return None, {"seed": seed, "reason": "completed-save markers missing", "missing": missing}
    hashes_before = {name: file_hash(directory/name) for name in names}
    trace_info = json.loads((directory/"trace-info.json").read_text(encoding="utf-8"))
    future_info = json.loads((directory/"future-info.json").read_text(encoding="utf-8"))
    placements = json.loads((directory/"fixed-placements.json").read_text(encoding="utf-8"))
    stored = torch.load(directory/"statistics.pt", map_location="cpu", weights_only=True)
    future = torch.load(directory/"future-statistics.pt", map_location="cpu", weights_only=True)
    if not isinstance(stored, dict) or set(stored) != {"calibration", "tuning"}:
        raise ValueError("statistics.pt must contain calibration and tuning only")
    inputs = {}
    for split, expected_windows in (("calibration",1024), ("tuning",512)):
        metadata = trace_info[split]
        if metadata["chunks"] != expected_windows:
            raise ValueError("protocol calibration/tuning chunk count changed")
        inputs[split] = integer_counts(stored[split], metadata["chunks"], metadata["accepted_dispatches"])
    inputs["future"] = integer_counts(future, future_info["chunks"], future_info["accepted_dispatches"])
    if future_info["chunks"] <= 0:
        raise ValueError("future trace is empty")
    hashes_after = {name: file_hash(directory/name) for name in names}
    if hashes_after != hashes_before:
        raise RuntimeError("source files changed while loading; not a verified completed save")
    proof = {"seed": seed, "directory": str(directory.resolve()), "source_file_sha256": hashes_before,
             "tensors": {"calibration": tensor_provenance(stored["calibration"]),
                         "tuning": tensor_provenance(stored["tuning"]), "future": tensor_provenance(future)},
             "integer_tensor_hashes": {split: tensor_provenance(value) for split,value in inputs.items()},
             "complete_save_verified_by_metadata_and_before_after_hashes": True,
             "future_labels_or_token_chunks_read": False}
    return (inputs, placements, proof), None


def necessary_bound(expert_load):
    if expert_load.ndim != 1 or expert_load.device.type != "cpu" or expert_load.dtype != torch.int64 or bool((expert_load < 0).any()):
        raise ValueError("nonnegative integral CPU expert-load vector required")
    total = int(expert_load.sum())
    maximum = int(expert_load.max())
    return {"total_dispatches": total, "max_expert_load": maximum,
            "average_device_load": total/4, "maximum_allowed_device_load": 5*total/16,
            "max_expert_over_average_device_load": 4*maximum/total if total else None,
            "necessary_max_expert_bound_passed": 16*maximum <= 5*total,
            "decision": "exact integer16*load<=5*total; no floating tolerance"}


def placement_loads(expert_load, assignments):
    if expert_load.shape != (8,) or expert_load.dtype != torch.int64 or expert_load.device.type != "cpu":
        raise ValueError("eight integral CPU expert loads required")
    if bool((expert_load < 0).any()) or assignments.ndim != 2 or assignments.shape[1] != 8:
        raise ValueError("invalid expert loads/assignments")
    if assignments.dtype != torch.int64 or assignments.device.type != "cpu" or bool(((assignments < 0)|(assignments >= 4)).any()):
        raise ValueError("device IDs0..3 required")
    membership = assignments[...,None] == torch.arange(4)
    if not bool((membership.sum(1) == 2).all()):
        raise ValueError("every labeled device must store exactly two experts")
    loads = (membership.to(torch.int64) * expert_load[None,:,None]).sum(1)
    if not bool((loads.sum(1) == expert_load.sum()).all()):
        raise ArithmeticError("placement changed accepted dispatch total")
    return loads


def layer_feasibility(split_loads, assignments):
    totals = {split: int(load.sum()) for split,load in split_loads.items()}
    device_loads = {split: placement_loads(load, assignments) for split,load in split_loads.items()}
    passed = {split: (16*load.max(1).values <= 5*totals[split]) for split,load in device_loads.items()}
    # All feasibility decisions are exact integers; floating ratios below are
    # descriptive minimax scores and never admit/reject an assignment.
    imbalance = {split: 4*load.max(1).values.double()/totals[split] if totals[split]
                 else torch.zeros(len(assignments),dtype=torch.float64) for split,load in device_loads.items()}
    result = []
    for scenario, splits in SCENARIOS.items():
        feasible = torch.stack([passed[split] for split in splits]).all(0)
        score = torch.stack([imbalance[split] for split in splits]).max(0).values
        best = int(score.argmin())
        indices = torch.nonzero(feasible).flatten()
        witness_index = int(indices[0]) if len(indices) else None
        necessary_ok = all(necessary_bound(split_loads[split])["necessary_max_expert_bound_passed"] for split in splits)
        reason = ("feasible_assignment_exists" if len(indices) else
                  "infeasible_already_by_single_expert_necessary_bound" if not necessary_ok else
                  "infeasible_by_exhaustive_pair_capacity_and_split_constraints")
        result.append({"scenario": scenario, "splits_checked_separately": list(splits),
                       "assignments_checked": len(assignments), "feasible_assignments": len(indices),
                       "all_single_expert_necessary_bounds_passed": necessary_ok, "reason": reason,
                       "minimum_achievable_worst_device_over_split_average": float(score[best]),
                       "minimax_assignment_diagnostic_only": assignments[best].tolist(),
                       "first_feasible_assignment_diagnostic_only": assignments[witness_index].tolist() if witness_index is not None else None,
                       "witness_device_loads_by_split": {split: device_loads[split][witness_index].tolist() for split in splits} if witness_index is not None else None,
                       "future_used_as_retrospective_oracle": "future" in splits,
                       "new_primary_placement_authorized": False})
    return result, passed, device_loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[101,202,303])
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    protocol = Path(__file__).with_name("H10_moe_placement.md")
    sources = (Path(__file__), protocol, REPO/"tests"/"test_h10_posthoc_feasibility.py")
    source_hashes = {}
    for source in sources:
        source_hashes[str(source.relative_to(REPO))] = file_hash(source)
        shutil.copyfile(source,args.out/(source.name+".source"))
    primary_snapshot = (args.source/"manifest.json").read_bytes()
    (args.out/"primary-manifest.json.source").write_bytes(primary_snapshot)
    primary_manifest = json.loads(primary_snapshot)
    manifest = {"state":"running", "hypothesis":"H10", "stage":"posthoc_exhaustive_load_feasibility",
                "posthoc":True, "primary_series_modified":False, "new_primary_placement_selected":False,
                "source_run":str(args.source.resolve()), "source_primary_manifest_snapshot_sha256":file_hash(args.out/"primary-manifest.json.source"),
                "source_primary_state_at_snapshot":primary_manifest.get("state"), "source_hashes":source_hashes,
                "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                "base_sha":BASE_SHA, "command":sys.argv, "python":platform.python_version(), "torch":torch.__version__,
                "device":"cpu", "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"], "cpu_threads":2,
                "gpu_calls":False, "requested_seeds":args.seeds, "completed_seeds":[], "skipped_seeds":[],
                "experts_per_layer":8, "labeled_devices":4, "experts_per_device_exactly":2, "sparse_layers":12,
                "enumeration_count_formula":"8!/(2!^4)=2520",
                "load_bound":"device accepted load<=1.25*that split's total accepted load/4",
                "exact_decision":"16*device_load<=5*split_total",
                "limitations":["retrospective oracle uses future statistics only for diagnostics; never a new primary pi",
                               "indivisible experts, exactly2 per device and independent per-layer capacity; no cross-layer constraint",
                               "no communication objective, migration cost or physical latency optimization",
                               "aggregate accepted loads per split, not per-window worst-case service load",
                               "zero accepted-load split admits every capacity-valid assignment; imbalance ratio is undefined"]}
    write_json(args.out/"manifest.json",manifest)
    summaries, feasibility_rows, bound_rows, stored_rows, details, provenance = [], [], [], [], [], []
    try:
        subprocess.run(["git","merge-base","--is-ancestor",BASE_SHA,"HEAD"],cwd=REPO,check=True)
        manifest["corrected_base_gate"] = "passed"
        assignments = balanced_placements()
        write_json(args.out/"enumerated-assignments.json",assignments.tolist())
        manifest["enumerated_assignments_sha256"] = file_hash(args.out/"enumerated-assignments.json")
        for seed in args.seeds:
            loaded, skipped = load_seed(args.source,seed)
            if skipped:
                manifest["skipped_seeds"].append(skipped)
                continue
            inputs, saved, proof = loaded
            provenance.append(proof)
            write_json(args.out/"input-provenance.json",provenance)
            loads = {split:tensor.sum((0,1)) for split,tensor in inputs.items()}  # [expert,layer]
            per_layer = []
            masks = torch.empty(12,len(assignments),3,dtype=torch.bool)
            for layer in range(12):
                vectors = {split:value[:,layer] for split,value in loads.items()}
                for split, vector in vectors.items():
                    bound_rows.append({"seed":seed,"sparse_layer_index":layer,"split":split,**necessary_bound(vector)})
                evidence, passed, _ = layer_feasibility(vectors,assignments)
                for split_index,split in enumerate(SPLITS):
                    masks[layer,:,split_index] = passed[split]
                for item in evidence:
                    detail = {"seed":seed,"sparse_layer_index":layer,**item}
                    per_layer.append(detail)
                    details.append(detail)
                    feasibility_rows.append({"seed":seed,"sparse_layer_index":layer,"scenario":item["scenario"],
                         "assignments_checked":item["assignments_checked"],"feasible_assignments":item["feasible_assignments"],
                         "minimum_worst_imbalance":item["minimum_achievable_worst_device_over_split_average"],
                         "single_expert_necessary_bound_passed":item["all_single_expert_necessary_bounds_passed"],
                         "reason":item["reason"],"first_feasible_witness_diagnostic_only":json.dumps(item["first_feasible_assignment_diagnostic_only"]),
                         "posthoc_future_oracle":item["future_used_as_retrospective_oracle"],"new_primary_pi":False})
                for method, placement in saved["placements"].items():
                    if placement is None:
                        continue
                    if len(placement) != 12:
                        raise ValueError("saved primary placement does not have12 sparse layers")
                    assignment = torch.tensor([placement[layer]],dtype=torch.int64)
                    placement_pass = {split:bool(16*placement_loads(vector,assignment).max() <= 5*int(vector.sum()))
                                      for split,vector in vectors.items()}
                    for item in evidence:
                        valid = all(placement_pass[split] for split in item["splits_checked_separately"])
                        diagnosis = ("stored_placement_feasible" if valid else
                                     "stored_placement_failed_but_constraint_system_feasible" if item["feasible_assignments"] else
                                     "constraint_system_infeasible_for_every_capacity_valid_placement")
                        stored_rows.append({"seed":seed,"method":method,"sparse_layer_index":layer,"scenario":item["scenario"],
                                            "stored_placement_passed":valid,"exact_feasible_assignments":item["feasible_assignments"],
                                            "diagnosis":diagnosis,"placement":json.dumps(placement[layer]),"posthoc":True})
            torch.save(masks,args.out/f"seed-{seed}-individual-split-feasibility.pt")
            summary = {"seed":seed,"posthoc":True,"new_primary_pi":False,"scenarios":{}}
            for scenario in SCENARIOS:
                entries = [item for item in per_layer if item["scenario"] == scenario]
                impossible = [item["sparse_layer_index"] for item in entries if not item["feasible_assignments"]]
                summary["scenarios"][scenario] = {"all12_layers_jointly_feasible":not impossible,
                    "infeasible_layers":impossible,"feasible_assignments_by_layer":{str(item["sparse_layer_index"]):item["feasible_assignments"] for item in entries},
                    "number_of_all_layer_assignments":math.prod(item["feasible_assignments"] for item in entries),
                    "future_oracle_only":"future" in SCENARIOS[scenario]}
            summaries.append(summary)
            manifest["completed_seeds"].append(seed)
            write_json(args.out/"summary.json",summaries)
            write_json(args.out/"layer-details.json",details)
            write_csv(args.out/"feasibility.csv",feasibility_rows)
            write_csv(args.out/"necessary-bounds.csv",bound_rows)
            write_csv(args.out/"stored-placement-audit.csv",stored_rows)
            manifest["peak_ram_bytes_sampled"] = max(manifest.get("peak_ram_bytes_sampled",0),psutil.Process().memory_info().rss)
            if time.perf_counter()-started > 600 or manifest["peak_ram_bytes_sampled"] > 24*1024**3:
                raise RuntimeError("bounded CPU audit time/RSS budget exceeded")
            print(json.dumps(summary),flush=True)
        if not manifest["completed_seeds"]:
            raise RuntimeError("no source seed has verified completed calibration/tuning/future saves")
        manifest["state"] = "completed_posthoc_feasibility_audit"
        write_json(args.out/"table-units.json",{"load":"accepted expert dispatches","imbalance":"device load / split average device load",
                   "feasible_assignments":"labeled exact2-per-device placements; integer count","future_oracle":"retrospective only"})
    except Exception as error:
        manifest.update(state="posthoc_audit_failed",error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter()-started
        write_json(args.out/"manifest.json",manifest)


if __name__ == "__main__":
    main()
