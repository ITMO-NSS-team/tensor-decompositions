"""Independent continuation, real four-dimensional alias and grid checks."""
import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision", reason="real neural experiment optional dependency")
from experiments.hypotheses import run_h09_real as h09


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class RealShapeStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.other = nn.Parameter(torch.tensor([.3, -.8]))
        self.layer3 = nn.ModuleList([nn.Module()])
        self.layer3[0].conv2 = nn.Conv2d(256,256,3,padding=1,bias=False)


def warm_model_and_state():
    model = RealShapeStateModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(9451)
    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.randn(parameter.shape, generator=rng)
        optimizer.step()
    return model, copy.deepcopy(optimizer.state_dict()), copy.deepcopy(optimizer.state)


def test_dense_continuation_matches_independent_adamw_state_and_next_step():
    warm, saved, _ = warm_model_and_state()
    dense = copy.deepcopy(warm)
    independent = copy.deepcopy(warm)
    actual, compressed = h09.continuation_optimizer(dense, saved, "dense", 64, 101)
    expected = torch.optim.AdamW(independent.parameters(), lr=.001, weight_decay=.0001)
    expected.load_state_dict(copy.deepcopy(saved))
    for group in expected.param_groups:
        group["lr"], group["weight_decay"] = .0001, .0001
    assert compressed is None
    for p, q in zip(dense.parameters(), independent.parameters()):
        assert float(actual.state[p]["step"]) == 3
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual.state[p][field], expected.state[q][field], atol=0, rtol=0)
        p.grad = torch.full_like(p, .17)
        q.grad = p.grad.clone()
    actual.step()
    expected.step()
    for p, q in zip(dense.parameters(), independent.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        assert float(actual.state[p]["step"]) == 4


@pytest.mark.parametrize("method", ("fixed", "adaptive", "fixed_paid"))
def test_compressed_target_excludes_dense_state_and_keeps_other_warm_moments(method):
    warm, saved, _ = warm_model_and_state()
    model = copy.deepcopy(warm)
    optimizer, compressed = h09.continuation_optimizer(model, saved, method, 64, 101)
    target = model.layer3[0].conv2.weight
    assert all(parameter is not target for group in optimizer.param_groups for parameter in group["params"])
    assert target not in optimizer.state
    other_state = optimizer.state[model.other]
    saved_other = saved["state"][saved["param_groups"][0]["params"][0]]
    for field in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(other_state[field], saved_other[field], atol=0, rtol=0)
    assert compressed.e.shape == (256,2304)
    assert compressed.rank == 64 and compressed.solver == "rsvd"
    assert compressed.q is compressed.m is compressed.v is None
    assert compressed.tau == compressed.updates == 0
    assert torch.count_nonzero(compressed.e) == 0
    # Loading is a copy, so subsequent continuation does not edit the shared
    # saved warm optimizer state used by the remaining paired methods.
    optimizer.state[model.other]["exp_avg"].zero_()
    assert torch.count_nonzero(saved_other["exp_avg"]) > 0


def test_real_4d_weight_step_matches_independent_coordinate_adam_formula():
    rng = torch.Generator().manual_seed(138)
    target = nn.Parameter(torch.randn(256,256,3,3,generator=rng))
    target.grad = torch.randn(target.shape, generator=rng)
    before = target.detach().clone()
    gradient = target.grad.clone()
    rank = 3
    q = torch.linalg.qr(torch.randn(256,rank,generator=rng)).Q
    matrix_gradient = gradient.reshape(256,2304)
    residual = .1 * torch.randn(256,2304,generator=rng)
    projected = q.T @ (matrix_gradient + residual)
    old_m, old_v, tau = torch.randn(rank,2304,generator=rng), torch.rand(rank,2304,generator=rng), 4
    compressed = h09.CompressedAdam((256,2304), "cpu", rank=rank, refresh="fixed", seed=101)
    compressed.q, compressed.e, compressed.m, compressed.v = q.clone(), residual.clone(), old_m.clone(), old_v.clone()
    compressed.tau, compressed.updates = tau, 1
    m = .9*old_m + .1*projected
    v = .999*old_v + .001*projected.square()
    direction = q @ ((m/(1-.9**(tau+1))) / (torch.sqrt(v/(1-.999**(tau+1)))+1e-8))
    expected = before.reshape(256,2304) * (1-.0001*.0001) - .0001 * direction
    h09.compressed_conv_step(target, compressed, 1)
    torch.testing.assert_close(target.reshape(256,2304), expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(target.grad, gradient, atol=0, rtol=0)
    torch.testing.assert_close(compressed.m, m, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compressed.v, v, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compressed.e + q @ projected, matrix_gradient + residual, atol=2e-6, rtol=2e-6)
    assert compressed.tau == 5 and compressed.updates == 1


def test_noncontiguous_conv_weight_cannot_silently_update_a_reshape_copy():
    target = nn.Parameter(torch.ones(256,256,3,3).transpose(1,2))
    target.grad = torch.ones_like(target)
    compressed = h09.CompressedAdam((256,2304), "cpu", rank=3)
    with pytest.raises(ValueError, match="share convolution storage"):
        h09.compressed_conv_step(target, compressed, 0)
    assert compressed.q is None


class GridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(x) * self.weight


def test_all_41_grid_checkpoints_are_actual_observed_states(tmp_path, monkeypatch):
    # Cheap independent shell check, isolating checkpoint/evaluation timing
    # from the separately checked real256x2304 optimizer adapter.
    warm = GridModel().eval()
    optimizer = torch.optim.SGD(warm.parameters(), lr=.01)
    monkeypatch.setattr(h09, "continuation_optimizer", lambda *a, **k: (torch.optim.SGD(a[0].parameters(), lr=.01), None))
    monkeypatch.setattr(h09, "sync", lambda: None)
    monkeypatch.setattr(h09, "guard", lambda *a, **k: None)
    monkeypatch.setattr(h09, "seed_all", lambda seed: torch.random.default_generator.manual_seed(seed))
    seen = []
    def observe(model, *args):
        value = float(model.weight.detach())
        seen.append(value)
        model.eval()
        return {"accuracy": .5, "cross_entropy": value, "n": 3}
    monkeypatch.setattr(h09, "evaluate", observe)
    inputs = torch.arange(8,dtype=torch.float32).reshape(4,2)/4
    labels = torch.tensor([0,1,0,1])
    data = {"augmented": TensorDataset(inputs,labels), "tuning": None}
    metadata, observations = h09.run_branch(warm, optimizer.state_dict(), data, {"recovery": list(range(4))},
                                             101, "dense", 64, "cpu", tmp_path, 400)
    assert [item["step"] for item in observations] == list(range(0,401,10))
    assert len(seen) == len(observations) == 41
    for observation, expected in zip(observations, seen):
        state = torch.load(observation["checkpoint"], map_location="cpu", weights_only=True)
        assert float(state["weight"]) == expected
        assert h09.sha(observation["checkpoint"]) == observation["checkpoint_sha256"]
    assert metadata["steps"] == 400
    assert len(metadata["batch_sha256"]) == 64
    assert metadata["batch_sha256"] != hashlib.sha256(b"").hexdigest()
