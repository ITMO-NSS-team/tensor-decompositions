"""Independent admission of optimizer-state retention and convolution storage."""
import importlib.util
from pathlib import Path
import copy
import pytest
import torch
from torch import nn

pytest.importorskip('torchvision')
path = Path(__file__).parents[1] / 'experiments/hypotheses/run_h09_real.py'
spec = importlib.util.spec_from_file_location('h09_real_continuation', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TinyShell(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.conv2 = nn.Conv2d(256, 256, 3, bias=False)
        self.layer3 = nn.ModuleList([block])
        self.head = nn.Linear(2, 2)


def test_compressed_optimizer_keeps_other_moments_and_excludes_target():
    model = TinyShell()
    warm = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    warm.step()
    state = copy.deepcopy(warm.state_dict())
    optimizer, compressed = runner.continuation_optimizer(model, state, 'fixed', 64, 101)
    target = model.layer3[0].conv2.weight
    assert all(parameter is not target for group in optimizer.param_groups for parameter in group['params'])
    assert target not in optimizer.state
    for parameter in model.head.parameters():
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[parameter][key], warm.state[parameter][key])
    assert compressed.tau == 0 and compressed.updates == 0
    assert compressed.e.shape == (256, 2304)
    assert torch.count_nonzero(compressed.e) == 0
    assert optimizer.param_groups[0]['lr'] == .0001
    assert state['param_groups'][0]['lr'] == .001


def test_dense_optimizer_retains_target_moments():
    model = TinyShell()
    warm = torch.optim.AdamW(model.parameters(), lr=.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    warm.step()
    optimizer, compressed = runner.continuation_optimizer(model, warm.state_dict(), 'dense', 64, 101)
    target = model.layer3[0].conv2.weight
    assert compressed is None
    torch.testing.assert_close(optimizer.state[target]['exp_avg_sq'], warm.state[target]['exp_avg_sq'])
    assert int(optimizer.state[target]['step']) == 1


def test_matrix_intervention_updates_same_conv_and_preserves_gradient():
    target = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2))
    target.grad = torch.full_like(target, 2.)
    before = target.detach().clone()
    gradient = target.grad.clone()
    class ReferenceStep:
        def step(self, matrix, step, lr):
            assert matrix.shape == (3, 8) and step == 7
            torch.testing.assert_close(matrix.grad, gradient.reshape(3, 8))
            matrix.add_(matrix.grad, alpha=-lr)
    runner.compressed_conv_step(target, ReferenceStep(), 7, lr=.01, weight_decay=.1)
    torch.testing.assert_close(target, before*.999-.01*gradient)
    torch.testing.assert_close(target.grad, gradient, rtol=0, atol=0)


def test_compressed_resume_matches_uninterrupted_next_step():
    from run_h09_synthetic import CompressedAdam
    generator=torch.Generator().manual_seed(812)
    parameter=nn.Parameter(torch.randn(9,5,generator=generator))
    optimizer=CompressedAdam(parameter.shape,'cpu',rank=3,refresh='fixed',seed=12)
    parameter.grad=torch.randn(parameter.shape,generator=generator)
    optimizer.step(parameter,0)
    restored_parameter=nn.Parameter(parameter.detach().clone())
    restored=CompressedAdam(parameter.shape,'cpu',rank=3,refresh='fixed',seed=12)
    restored.load_state_dict(optimizer.state_dict())
    gradient=torch.randn(parameter.shape,generator=generator)
    parameter.grad=gradient.clone();restored_parameter.grad=gradient.clone()
    optimizer.step(parameter,1);restored.step(restored_parameter,1)
    torch.testing.assert_close(restored_parameter,parameter,rtol=0,atol=0)
    for name in ('e','q','m','v'):
        torch.testing.assert_close(getattr(restored,name),getattr(optimizer,name),rtol=0,atol=0)
    assert restored.tau==optimizer.tau==2


def test_effective_batch_preserves_epoch_tail_before_next_shuffle():
    indices=list(range(20000,25000))
    batches=runner.fixed_batches(indices,41,101)
    assert batches.shape==(41,128)
    flattened=batches.flatten().tolist()
    assert set(flattened[:5000])==set(indices) and len(set(flattened[:5000]))==5000
    assert all(index in indices for index in flattened)
    torch.testing.assert_close(batches,runner.fixed_batches(indices,41,101),rtol=0,atol=0)
