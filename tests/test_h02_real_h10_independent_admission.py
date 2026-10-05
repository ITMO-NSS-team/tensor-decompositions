"""Independent graph, BN and capacity/load admissions for new real/toy runners."""
import copy

import pytest
import torch
pytest.importorskip('torchvision')
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

from experiments.hypotheses.real_composition_core import ComposedBlock
from experiments.hypotheses.run_h02_real import common_bn
from experiments.hypotheses import run_h10_synthetic as h10


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_h02_full_rank_preserves_learned_bn_and_residual_graph():
    torch.manual_seed(92)
    original = BasicBlock(8, 8).double().eval()
    with torch.no_grad():
        for bn in (original.bn1, original.bn2):
            bn.weight.copy_(torch.linspace(.6, 1.4, 8, dtype=torch.float64))
            bn.bias.copy_(torch.linspace(-.15, .15, 8, dtype=torch.float64))
            bn.running_mean.copy_(torch.linspace(-.3, .3, 8, dtype=torch.float64))
            bn.running_var.copy_(torch.linspace(.5, 1.7, 8, dtype=torch.float64))
    student = ComposedBlock(original, 8).eval()
    x = torch.randn(3, 8, 2, 2, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(student(x), original(x), rtol=1e-10, atol=1e-10)
    a, = torch.autograd.grad(student(x).square().sum(), x)
    b, = torch.autograd.grad(original(x).square().sum(), x)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_h02_shared_initial_is_equal_but_independently_trainable_and_bn_frozen():
    torch.manual_seed(93)
    block = ComposedBlock(BasicBlock(8, 8).eval(), 3, shared_initialization=True).eval()
    torch.testing.assert_close(block.q1, block.q2, rtol=0, atol=0)
    before = block.q2.detach().clone()
    with torch.no_grad():
        block.q1[0, 0].add_(1)
    torch.testing.assert_close(block.q2, before, rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in block.factors()) == 20 * 8 * 3
    assert all(not parameter.requires_grad for bn in (block.bn1, block.bn2) for parameter in bn.parameters())
    block(torch.randn(3, 8, 2, 2)).square().sum().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in block.factors())
    assert all(parameter.grad is None for bn in (block.bn1, block.bn2) for parameter in bn.parameters())


def test_h02_common_bn_changes_only_two_target_bn_buffers():
    torch.manual_seed(94)
    teacher = resnet18(weights=None, num_classes=10).eval()
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    common_bn(teacher, [(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))], "cpu")
    changed = {name for name, value in teacher.state_dict().items() if not torch.equal(value, before[name])}
    assert changed
    allowed = {f"layer3.1.bn{index}.{buffer}" for index in (1, 2)
               for buffer in ("running_mean", "running_var", "num_batches_tracked")}
    assert changed <= allowed
    assert all(not module.training for module in teacher.modules())


def test_h10_greedy_filters_load_before_cheapest_destination():
    # Bound6.25; second expert cannot follow first to device0.
    counts = torch.zeros(1, 2, 4, 1)
    counts[0, 0, :, 0] = torch.tensor([4., 3., 2., 1.])
    placement, feasible, loads = h10.place(counts, "cost")
    assert feasible
    assert placement == [0, 1, 0, 1]
    assert loads == [6., 4.]


def test_h10_dispatch_cost_and_migration_match_independent_integer_count():
    model = h10.ToyMoE().eval()
    pair = h10.data(11, "cpu")["calibration"]
    counts = h10.trace(model, pair)
    chosen, _ = model.routes(pair[0])
    destinations = torch.tensor([1, 0, 1, 0])[chosen]
    remote = int((destinations != pair[2][:, None]).sum())
    result = h10.price(counts, [1, 0, 1, 0], [0, 1, 0, 1])
    assert result["remote_dispatches"] == remote
    assert result["exchange_bytes"] == remote * 256
    expert_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.experts[0].parameters())
    assert result["migration_bytes"] == 4 * expert_bytes
    assert int(counts.sum()) == 2 * len(pair[0])
