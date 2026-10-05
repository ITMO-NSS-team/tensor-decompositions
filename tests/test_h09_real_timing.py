"""Independent CPU accounting of continuation setup and durable checkpoints."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.hypotheses import run_h09_real as h09


class VirtualClock:
    def __init__(self):
        self.now = 100.0
        self.events = []

    def read(self):
        self.events.append("read")
        return self.now

    def charge(self, event, seconds):
        self.events.append(event)
        self.now += seconds


def zero_step_branch(directory, monkeypatch, method):
    """Use real serialization, charging named operations without wall-clock waits."""
    clock = VirtualClock()
    warm = nn.Linear(2, 2)
    original_copy = copy.deepcopy
    original_save = torch.save
    original_sha = h09.sha
    original_json = h09.write_json
    optimizer_seconds = 7 if method == "dense" else 17

    def clone(model):
        clock.charge("model_copy", 5)
        return original_copy(model)

    def initialize(model, *args):
        clock.charge("optimizer_init", optimizer_seconds)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        compressed = None if method == "dense" else SimpleNamespace(
            updates=0, events=[], state_dict=lambda: {"initial_state": True})
        return optimizer, compressed

    def batches(*args):
        clock.charge("batch_schedule", 11)
        return torch.empty(0, 128, dtype=torch.int64)

    def save(value, path):
        name = path.name
        cost = 13 if name == "batch-indices.pt" else 31 if name == "resume.pt" else 19
        clock.charge("save:" + name, cost)
        return original_save(value, path)

    def digest(path):
        cost = 41 if path.name == "batch-indices.pt" else 37 if path.name == "resume.pt" else 23
        clock.charge("sha:" + path.name, cost)
        return original_sha(path)

    def observe(model, *args):
        clock.charge("tuning_evaluation", 17)
        return {"accuracy": .75, "cross_entropy": .5, "n": 2}

    def persist(path, value):
        clock.charge("json:" + path.name, 29)
        return original_json(path, value)

    monkeypatch.setattr(h09, "time", SimpleNamespace(perf_counter=clock.read))
    monkeypatch.setattr(h09, "copy", SimpleNamespace(deepcopy=clone))
    monkeypatch.setattr(h09, "continuation_optimizer", initialize)
    monkeypatch.setattr(h09, "fixed_batches", batches)
    monkeypatch.setattr(h09.torch, "save", save)
    monkeypatch.setattr(h09, "sha", digest)
    monkeypatch.setattr(h09, "evaluate", observe)
    monkeypatch.setattr(h09, "write_json", persist)
    monkeypatch.setattr(h09, "sync", lambda: clock.events.append("sync"))
    monkeypatch.setattr(h09, "seed_all", lambda seed: None)
    metadata, observations = h09.run_branch(
        warm, {}, {"tuning": None}, {"recovery": [0, 1]},
        101, method, 64, "cpu", directory, 0)
    return metadata, observations, clock, 5 + optimizer_seconds + 11 + 13


@pytest.mark.parametrize("method", h09.METHODS)
def test_quality_time_includes_setup_and_current_checkpoint_hash(tmp_path, monkeypatch, method):
    metadata, observations, clock, setup = zero_step_branch(tmp_path, monkeypatch, method)
    assert len(observations) == 1 and observations[0]["step"] == 0
    # Even an initially qualifying checkpoint pays initialization, verification
    # of quality, model persistence and that checkpoint's integrity hash.
    assert observations[0]["elapsed_seconds"] == setup + 17 + 19 + 23
    assert metadata["setup_seconds"] == setup
    assert clock.events.index("read") < clock.events.index("model_copy")
    expected_hash = hashlib.sha256((tmp_path / "step-000.pt").read_bytes()).hexdigest()
    assert observations[0]["checkpoint_sha256"] == expected_hash


@pytest.mark.parametrize("method", ("dense", "adaptive"))
def test_branch_total_pays_resume_and_index_hashes_before_reporting(tmp_path, monkeypatch, method):
    metadata, observations, clock, setup = zero_step_branch(tmp_path, monkeypatch, method)
    # Current observation JSON is outside its own boundary but inside total.
    # Final projection/branch report serialization is explicitly outside total.
    expected = setup + 17 + 19 + 23 + 29 + 31 + 41 + 37
    assert metadata["seconds"] == expected
    reporting = 29 if method == "dense" else 58
    assert clock.now - 100 == expected + reporting
    persisted = json.loads((tmp_path / "branch.json").read_text())
    assert persisted["seconds"] == expected
    assert persisted["timing_scope"]["common_warmup_excluded"] is True
    assert persisted["timing_scope"]["final_reporting_excluded"] is True
