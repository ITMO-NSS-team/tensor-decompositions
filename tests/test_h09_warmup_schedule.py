"""Warmup must retain constant lr and usable Adam state, unlike baseline decay."""
import importlib.util
from pathlib import Path
import pytest
import torch

pytest.importorskip('torchvision')
path = Path(__file__).parents[1] / 'experiments/hypotheses/train_cifar_baseline.py'
spec = importlib.util.spec_from_file_location('h09_warmup_schedule', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_warmup_retains_lr_moments_and_scheduler_round_trip():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=.001)
    count, scheduler = runner.training_schedule(optimizer, True)
    rates = []
    for _ in range(count):
        parameter.grad = torch.tensor([1., -2.])
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    assert count == 5 and rates == [.001] * 5
    clone = torch.nn.Parameter(parameter.detach().clone())
    restored = torch.optim.AdamW([clone], lr=.001)
    _, restored_scheduler = runner.training_schedule(restored, True)
    restored.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    torch.testing.assert_close(restored.state[clone]['exp_avg'], optimizer.state[parameter]['exp_avg'])
    assert int(restored.state[clone]['step']) == 5
    restored_scheduler.step()
    assert restored.param_groups[0]['lr'] == .001


def test_default_baseline_keeps_cosine_thirty_epochs():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=.001)
    count, scheduler = runner.training_schedule(optimizer)
    assert count == 30
    for _ in range(count):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0.
