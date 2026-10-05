import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from experiments.hypotheses import run_h12_real as h12


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(x)))


class TinyTwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 1)
        self.layer2 = nn.Sequential(nn.Identity(), TinyBlock(4))
        self.layer3 = nn.Sequential(TinyBlock(4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.layer3(self.layer2(self.stem(x)))
        return self.head(x.mean((2, 3)))


def test_prespecified_real_pairs_and_fractional_channels_keep_spatial_full():
    assert h12.FUNCTIONAL_PAIRS == (((16,16),(32,32)),((32,32),(32,32)),((32,32),(64,64)),
                                    ((64,64),(32,32)),((64,64),(64,64)),((96,96),(96,96)))
    candidates = h12.rule_candidates("fractional")
    assert len(candidates) == 6
    for fraction, (pair, evidence) in zip(h12.FRACTIONS, candidates):
        assert pair == tuple((math.ceil(channels * fraction),) * 2 for channels in (128, 256))
        assert evidence["fraction"] == fraction
    assert h12.storage_count(((16,16),(32,32))) == (128*32+9*16**2)+(256*64+9*32**2)


def test_energy_rank_uses_original_diagonal_spectra_not_toy_grid():
    weights = torch.diag(torch.linspace(1., .2, 40)).reshape(40, 40, 1, 1).expand(-1, -1, 3, 3).clone()
    spectra = h12.original_channel_spectra(weights)
    epsilon = .3
    ranks, evidence = h12.energy_ranks(spectra, epsilon)
    singular_squared = 9 * torch.linspace(1., .2, 40).double().square()
    independent_budget = epsilon**2 * float(weights.double().square().sum()) / 4
    expected = next(r for r in range(1, 41) if float(singular_squared[r:].sum()) <= independent_budget)
    assert ranks == (expected, expected)
    assert expected > 16
    for rank, tail in zip(ranks, evidence["channel_tail_squared"]):
        assert tail <= evidence["per_mode_squared_budget"]
        assert spectra["modes"][0]["tail_squared"][rank-1] > evidence["per_mode_squared_budget"]
    candidates = h12.rule_candidates("energy", [spectra, spectra])
    assert len(candidates) == 6
    assert all(1 <= r <= 40 for pair, _ in candidates for ranks in pair for r in ranks)


def test_constant_effective_batches_preserve_permuted_epochs_and_pairing():
    batches = h12.constant_batch_indices(2000, 32, 173)
    assert batches.shape == (32, 128)
    flat = batches.flatten()
    assert sorted(flat[:2000].tolist()) == list(range(2000))
    assert sorted(flat[2000:4000].tolist()) == list(range(2000))
    assert torch.equal(batches, h12.constant_batch_indices(2000, 32, 173))
    assert not torch.equal(batches, h12.constant_batch_indices(2000, 32, 174))


def test_exact_reserve_preserves_outputs_and_frozen_original_bn_without_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("independent CPU admission must not query CUDA")
    for name in ("is_available", "current_device", "mem_get_info", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    h12.seed_all(934, "cpu")
    teacher = TinyTwoLayerModel().double().eval()
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.copy_(torch.linspace(-.1, .1, 4))
            module.running_var.copy_(torch.linspace(.5, 2, 4))
            module.weight.data.copy_(torch.linspace(.8, 1.3, 4))
            module.bias.data.copy_(torch.linspace(-.2, .2, 4))
    student, residuals = h12.make_candidate(teacher, ((4,4),(4,4)), 934)
    x = torch.randn(3, 1, 5, 5, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_(True)
    torch.testing.assert_close(student(x), teacher(xx), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(student(x).sum(), x)[0],
                               torch.autograd.grad(teacher(xx).sum(), xx)[0], atol=1e-11, rtol=1e-11)
    assert all(item["direct_weight_absolute_error"] == 0 for item in residuals)
    assert not student.training
    original = dict(teacher.named_buffers())
    for name, buffer in student.named_buffers():
        assert torch.equal(buffer, original[name])
    for name, parameter in student.named_parameters():
        assert parameter.requires_grad == any(name.startswith(target + ".") for target in h12.TARGETS)
    h12.ResourceGuard("cpu").check()


def test_restore_selected_factors_avoids_refactorization_and_preserves_shapes(monkeypatch):
    teacher = TinyTwoLayerModel().eval()
    candidate, _ = h12.make_candidate(teacher, ((2,2),(3,3)), 31)
    state = copy.deepcopy(candidate.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("selected/checkpoint restoration cannot rerun HOOI")
    monkeypatch.setattr(h12, "HOOIDecomposition", forbidden)
    restored = h12.restore_candidate(teacher, state)
    x = torch.randn(3, 1, 5, 5)
    torch.testing.assert_close(restored(x), candidate(x), atol=0, rtol=0)
    assert h12.layer_at(restored, h12.TARGETS[0]).core.shape == (2,2,3,3)
    assert h12.layer_at(restored, h12.TARGETS[1]).core.shape == (3,3,3,3)


def test_training_updates_both_targets_only_and_keeps_bn_and_effective_batch():
    h12.seed_all(511, "cpu")
    teacher = TinyTwoLayerModel().eval()
    student, _ = h12.make_candidate(teacher, ((2,2),(2,2)), 511)
    before = copy.deepcopy(student.state_dict())
    inputs = torch.randn(137, 1, 5, 5)
    targets = torch.randint(0, 2, (137,))
    dataset = TensorDataset(inputs, targets)
    phase, optimizer = h12.train_updates(student, dataset, list(range(137)), 99, "cpu", 2,
                                         h12.ResourceGuard("cpu"), microbatch=64)
    assert phase["steps"] == 2
    assert all(row["effective_batch"] == 128 for row in phase["history"])
    for name, value in student.state_dict().items():
        if not any(name.startswith(target + ".") for target in h12.TARGETS):
            assert torch.equal(value, before[name]), name
    assert all(not torch.equal(h12.layer_at(student, target).core, before[target + ".core"]) for target in h12.TARGETS)
    _, continued = h12.train_updates(student, dataset, list(range(137)), 100, "cpu", 1,
                                     h12.ResourceGuard("cpu"), optimizer=optimizer)
    assert continued is optimizer
    assert all(float(state["step"]) == 3 for state in optimizer.state.values())


def test_forecast_selection_checks_quality_before_cost_and_visible_reserve():
    candidates = [{"index": 0, "pair": ((16,16),(32,32)), "admissible": False, "forecast_cost_without_common_search": .01},
                  {"index": 1, "pair": ((32,32),(32,32)), "admissible": True, "forecast_cost_without_common_search": .02},
                  {"index": 2, "pair": ((64,64),(64,64)), "admissible": True, "forecast_cost_without_common_search": .03}]
    assert h12.choose_index(candidates) == 1
    assert h12.choose_index([{**item, "admissible": False} for item in candidates]) is None
    assert h12.initial_steps({"full_rank_reserve": True}) == 0
    assert h12.initial_steps({"full_rank_reserve": False}) == 32


def test_paired_cost_summary_keeps_three_training_units_and_no_acceptance_claim():
    rows, final = [], []
    for seed in h12.SEEDS:
        for rule, price in (("functional", .9), ("fractional", 1.), ("energy", 1.2)):
            rows.append({"seed": seed, "rule": rule, "measured_cost_seconds_per_image": price})
            final.append({"seed": seed, "rule": rule, "quality_passed": True})
    result = h12.paired_cost_summary(rows, final)
    assert result[0]["n_seeds"] == 3
    assert result[0]["mean_relative_cost_reduction"] == pytest.approx(.1)
    assert result[1]["mean_relative_cost_reduction"] == pytest.approx(.25)
    assert all(item["all_final_quality_guards_passed"] for item in result)
    assert all("no automatic hypothesis verdict" in item["scope"] for item in result)


def test_admission_and_latency_are_cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU admission cannot query CUDA")
    for name in ("is_available", "current_device", "mem_get_info", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert h12.admission()["gpu_calls"] is False
    model = TinyTwoLayerModel().eval()
    timing = h12.measured_latency(model, torch.ones(1,1,5,5), "cpu", h12.ResourceGuard("cpu"), warmups=1, repeats=2)
    assert timing["batch"] == 1 and timing["measured"] == 2
