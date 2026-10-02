import copy
import math

import pytest
import torch

from tdecomp.grad_proj.tensorgrad import AdamW, ParallelTG, ULTG
from tdecomp.grad_proj.tensorgrad.projectors.galore_projector import GaLoreProjector
from tdecomp.grad_proj.tensorgrad.projectors.tensor_unstructured_sparse_projector import TensorGradUnstructuredProjector


def run_projected_steps(preset, rank, **kwargs):
    model = torch.nn.Linear(7, 5, bias=True).double()
    optimizer, scheduler = preset(model, "truncated_svd", rank, **kwargs)
    for step in range(2):
        gradient = torch.arange(1, 36, dtype=torch.float64).reshape(5, 7) + step
        model.weight.grad = gradient.clone()
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()
        scheduler.step()
        torch.testing.assert_close(model.weight.grad, gradient, rtol=0, atol=0)
    state = optimizer.state[model.weight]
    lowrank_prefix = "first" if preset is ParallelTG else "second"
    sparse_prefix = "second" if preset is ParallelTG else "first"
    assert isinstance(state[lowrank_prefix + "_proj"], GaLoreProjector)
    assert isinstance(state[sparse_prefix + "_proj"], TensorGradUnstructuredProjector)
    assert torch.isfinite(model.weight).all()
    return state, lowrank_prefix, sparse_prefix


@pytest.mark.parametrize("preset", [ParallelTG, ULTG])
@pytest.mark.parametrize("rank, expected_rank", [(2, 2), (0.6, 3)])
@pytest.mark.parametrize("container", [tuple, list])
def test_experiment_pair_uses_lowrank_rank_and_sparse_ratio(preset, rank, expected_rank, container):
    state, lowrank, sparse = run_projected_steps(preset, container((rank, 0.4)))
    assert state[lowrank + "_exp_avg"].shape == (expected_rank, 7)
    assert state[lowrank + "_proj"].ortho_matrix.shape == (5, expected_rank)
    assert state[sparse + "_proj"]._indices.numel() == 14
    assert state[sparse + "_exp_avg"].numel() == 14


@pytest.mark.parametrize("preset, ratio", [(ParallelTG, 0.25), (ULTG, 0.1)])
@pytest.mark.parametrize("rank", [2, (2,), [2]])
def test_scalar_rank_keeps_sparse_default(preset, ratio, rank):
    state, lowrank, sparse = run_projected_steps(preset, rank)
    assert state[lowrank + "_exp_avg"].shape == (2, 7)
    assert state[sparse + "_proj"]._indices.numel() == math.ceil(35 * ratio)


@pytest.mark.parametrize("preset, options", [
    (ParallelTG, {"second_sparse_ratio": 0.6}),
    (ULTG, {"sparse_ratio": 0.6, "second_rank": 3}),
])
def test_explicit_branch_options_override_pair(preset, options):
    state, lowrank, sparse = run_projected_steps(preset, (2, 0.4), **options)
    assert state[lowrank + "_exp_avg"].shape == (3 if preset is ULTG else 2, 7)
    assert state[sparse + "_proj"]._indices.numel() == 21


@pytest.mark.parametrize("preset", [ParallelTG, ULTG])
@pytest.mark.parametrize("rank", [(), (2, 0.4, 0.5), (2, 0), (2, -0.1), (2, 1.1), (2, True), (2, float("nan"))])
def test_invalid_experiment_pair_rejected_before_model_changes(preset, rank):
    model = torch.nn.Linear(7, 5).double()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        preset(model, "truncated_svd", rank)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("legacy_signature", [False, True])
def test_adamw_factory_matches_torch_baseline_and_forwards_options(legacy_signature):
    model = torch.nn.Linear(7, 5).double()
    reference = copy.deepcopy(model)
    options = dict(learning_rate=0.02, weight_decay=0.03, betas=(0.8, 0.95), eps=1e-4,
                   scheduler="StepLR", gamma=0.5, step_size=1)
    arguments = ("truncated_svd", (2, 0.4)) if legacy_signature else ()
    optimizer, scheduler = AdamW(model, *arguments, **options)
    assert type(optimizer) is torch.optim.AdamW
    baseline = torch.optim.AdamW(reference.parameters(), lr=0.02, weight_decay=0.03,
                                betas=(0.8, 0.95), eps=1e-4)
    baseline_scheduler = torch.optim.lr_scheduler.StepLR(baseline, step_size=1, gamma=0.5)
    for step in range(3):
        for actual, expected in zip(model.parameters(), reference.parameters()):
            gradient = torch.arange(actual.numel(), dtype=actual.dtype).reshape(actual.shape) * 0.1 + step
            actual.grad, expected.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        baseline.step()
        scheduler.step()
        baseline_scheduler.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert scheduler.get_last_lr() == baseline_scheduler.get_last_lr()
    assert all("first_proj" not in state for state in optimizer.state.values())
