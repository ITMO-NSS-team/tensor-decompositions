"""Independent numerical and protocol checks for the H02 synthetic entry point."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import tensorly as tl
import torch

from experiments.hypotheses import run_h02_synthetic as h02


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_admission_includes_full_graph_gradients_svd_and_relu_counterexample():
    result = h02.admission_checks()
    for name in ("identity", "relu", "silu"):
        assert result[f"full_width_identity_graph_{name}"] == "passed_output_and_gradients"
    assert result["separate_svd_reference_and_parameter_counts"] == "passed"
    assert result["relu_movement_counterexample_norm"] == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize("control,expected_overlap", [("aligned", 8), ("rotated", 0), ("flat", 8)])
def test_teacher_shapes_spectra_rotation_and_repeatability(control, expected_overlap):
    teacher, splits, recipe = h02.synthetic_data(11, control=control, dtype=torch.float64, include_test=False)
    assert sum(p.numel() for p in teacher.parameters()) == 4224
    assert set(splits) == {"recovery", "calibration", "tuning"}
    assert [len(splits[name][0]) for name in ("recovery", "calibration", "tuning")] == [4096, 512, 512]
    assert recipe["leading_overlap_frobenius_squared"] == pytest.approx(expected_overlap, abs=1e-12)
    expected = torch.tensor([1.] * (32 if control == "flat" else 8)
                            + ([] if control == "flat" else [.3] * 24), dtype=torch.float64)
    torch.testing.assert_close(torch.linalg.svdvals(teacher.w1), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.svdvals(teacher.w2), expected, atol=1e-12, rtol=1e-12)
    repeated, other, _ = h02.synthetic_data(11, control=control, dtype=torch.float64, include_test=False)
    for left, right in zip(teacher.parameters(), repeated.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name in splits:
        assert h02.tensor_hash(splits[name][0]) == h02.tensor_hash(other[name][0])
        torch.testing.assert_close(teacher(splits[name][0]), splits[name][1], rtol=0, atol=0)
    assert len({h02.tensor_hash(pair[0]) for pair in splits.values()}) == 3


def test_separate_svd_starts_are_identical_and_common_q_is_smaller():
    teacher, _, _ = h02.synthetic_data(11, include_test=False)
    separate = h02.initialize(teacher, 8, "independent")
    joint = h02.initialize(teacher, 8, "joint")
    for key, value in separate.state_dict().items():
        torch.testing.assert_close(value, joint.state_dict()[key], rtol=0, atol=0)
    assert [tuple(p.shape) for p in separate.parameters()] == [(8, 32), (64, 8), (64, 8), (32, 8)]
    assert sum(p.numel() for p in separate.parameters()) == 1536
    shared = h02.initialize(teacher, 8, "shared_q")
    assert shared.q1 is shared.q2
    assert sum(p.numel() for p in shared.parameters()) == 1024
    assert "head" in dict(separate.named_buffers())
    assert "head" not in dict(separate.named_parameters())


def test_standalone_svd_uses_torch_without_changing_tensorly_backend():
    teacher, _, _ = h02.synthetic_data(11, include_test=False)
    with tl.backend_context("numpy"):
        student = h02.initialize(teacher, 8, "joint")
        assert tl.get_backend() == "numpy"
        assert all(isinstance(parameter, torch.Tensor) for parameter in student.parameters())


def test_nonlinearity_observes_original_width(monkeypatch):
    teacher, pairs, _ = h02.synthetic_data(11, include_test=False)
    student = h02.initialize(teacher, 4, "joint")
    seen = []
    original = h02.activation

    def observe(x, name):
        seen.append(tuple(x.shape))
        return original(x, name)

    monkeypatch.setattr(h02, "activation", observe)
    prediction = student(pairs["tuning"][0][:5])
    assert prediction.shape == (5, 4)
    assert seen == [(5, 64)]


def test_fitting_budget_targets_batch_order_head_and_teacher_ownership():
    teacher, pairs, _ = h02.synthetic_data(11, include_test=False)
    teacher_before = copy.deepcopy(teacher.state_dict())
    batches = h02.minibatches(50011, 512, 4, "cpu")
    assert h02.tensor_hash(batches) == h02.tensor_hash(h02.minibatches(50011, 512, 4, "cpu"))
    histories = {}
    for method in ("independent", "joint"):
        student = h02.initialize(teacher, 8, method)
        histories[method] = h02.train_phase(student, teacher, pairs["calibration"], batches,
                                             method=method, phase="calibration")
        recovery = h02.train_phase(student, teacher, pairs["recovery"], h02.minibatches(60011, 4096, 3, "cpu"),
                                  method=method, phase="recovery")
        assert recovery["steps"] == 3
        assert recovery["targets"] == ["network_output"] * 3
        torch.testing.assert_close(student.head, teacher.head, rtol=0, atol=0)
    assert histories["independent"]["targets"] == ["first_layer_output"] * 2 + ["second_layer_output_on_teacher_hidden"] * 2
    assert histories["joint"]["targets"] == ["block_output"] * 4
    assert histories["independent"]["minibatches_sha256"] == histories["joint"]["minibatches_sha256"]
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="even update count"):
        h02.train_phase(student, teacher, pairs["calibration"], batches[:3], method="independent", phase="calibration")


def test_zero_target_error_is_finite_and_explicitly_absolute():
    error, kind = h02.normalized_error(torch.ones(2, 4), torch.zeros(2, 4))
    assert error == 1
    assert kind == "absolute_mse_zero_target"
    assert h02.normalized_error(torch.zeros(2, 4), torch.zeros(2, 4)) == (0, kind)


def test_cuda_guard_resolves_device_index_without_accessing_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def indexed(device):
        assert isinstance(device, torch.device) and device.type == "cuda" and device.index == 0
        calls.append(device)
        return 0

    for name in ("reset_peak_memory_stats", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, indexed)

    def info(device):
        indexed(device)
        return 16 * 1024**3, 16 * 1024**3

    monkeypatch.setattr(torch.cuda, "mem_get_info", info)
    guard = h02.ResourceGuard("cuda", max_seconds=60)
    assert guard.device == torch.device("cuda:0")
    assert guard.metrics()["peak_gpu_bytes"] == 0
    assert calls


def test_primary_summary_does_not_count_methods_batches_or_incomplete_seeds_as_replications():
    rows = []
    for seed in h02.SEEDS:
        for method, value in (("independent", 1.0), ("joint", 0.8), ("shared_q", 0.01)):
            rows.append({"seed": seed, "method": method, "status": "complete", "block_nmse_before_recovery": value})
    summary = h02.paired_summary(rows)[0]
    assert summary["n_seeds"] == 5
    assert summary["mean_delta"] == pytest.approx(0.2)
    assert summary["ci95_low"] == pytest.approx(0.2)
    assert summary["outcome"] == "supported_primary_synthetic_only"
    summary = h02.paired_summary(rows[:3])[0]
    assert summary["n_seeds"] == 1
    assert summary["ci95_low"] is None
    assert summary["outcome"] == "indeterminate"
    rows[1]["status"] = "stopped"
    assert h02.paired_summary(rows)[0]["outcome"] == "indeterminate"


def test_pilot_cpu_run_never_queries_cuda_or_creates_final_test(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU pilot must not query or synchronize CUDA")

    for name in ("synchronize", "reset_peak_memory_stats", "mem_get_info", "memory_reserved",
                 "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    args = SimpleNamespace(device="cpu", activation="relu", control="aligned", mode="pilot", rank=8,
                           max_seconds=60, methods=("joint",))
    rows = h02.run_seed(tmp_path, 11, args)
    assert len(rows) == 1
    assert all(row["status"] == "pilot_complete" for row in rows)
    assert all(row["calibration_steps"] == 20 and row["recovery_steps"] == 0 for row in rows)
    inputs = json.loads((tmp_path / "seed-11" / "inputs.json").read_text())
    assert "test" not in inputs
    assert not list(tmp_path.rglob("final_test.json"))
    joint = json.loads((tmp_path / "seed-11" / "joint" / "history.json").read_text())
    assert joint["calibration"]["steps"] == 20


def test_confirmation_cannot_change_preregistered_seeds_and_args_reject_duplicates(tmp_path):
    default = h02.parse_args(["--out", str(tmp_path)])
    assert default.seeds == (11,) and default.methods == ("joint",)
    with pytest.raises(SystemExit):
        h02.parse_args(["--out", str(tmp_path), "--mode", "confirm", "--seeds", "123"])
    with pytest.raises(SystemExit):
        h02.parse_args(["--out", str(tmp_path), "--seeds", "11,11"])
    with pytest.raises(SystemExit):
        h02.parse_args(["--out", str(tmp_path), "--methods", "joint,independent"])
