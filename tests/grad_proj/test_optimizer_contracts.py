import copy
import io
import math
import random
from itertools import combinations, product
import numpy as np
import pytest
import torch
import tensorly as tl
from tdecomp.grad_proj.tensorgrad import TensorGRaD, ParallelTG, ULTG, setup_optimizer_and_scheduler
from tdecomp.grad_proj.tensorgrad.config import DataConfig, OptimizerConfig, TensorGRaDConfig
from tdecomp.grad_proj.tensorgrad.projectors.galore_projector import GaLoreProjector
from tdecomp.grad_proj.tensorgrad.projectors.tensor_lowrank_projector import TensorGradLowRankProjector
from tdecomp.grad_proj.tensorgrad.projectors.tensor_sparse_projector import TensorGradSparseProjector
from tdecomp.grad_proj.tensorgrad.projectors.tensor_unstructured_sparse_projector import TensorGradUnstructuredProjector
from tdecomp.grad_proj.tensorgrad.projectors.sparse_projector import GaLoreSparseProjector
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler, UPDATE_MODES
from tdecomp.grad_proj.tensorgrad.projectors.projector_utils import projector_from_state
from tdecomp.grad_proj.tensorgrad.training_utils.get_scheduler import get_scheduler, SCHEDULER_NAMES
from tdecomp.grad_proj.tensorgrad.training_utils.training_state import save_training_state, load_training_state


class Weights(torch.nn.Module):
    def __init__(self, shape=(4, 5), dtype=torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, math.prod(shape) + 1, dtype=torch.float64).reshape(shape).to(dtype))
        self.bias = torch.nn.Parameter(torch.ones(2, dtype=dtype))


@pytest.mark.parametrize("cls", [TensorGradSparseProjector, TensorGradUnstructuredProjector])
@pytest.mark.parametrize("method", ["topk", "randk", "probability", "randomk", "probablility"])
@pytest.mark.parametrize("shape", [(3, 4), (4, 5, 6), (1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_selection_and_reconstruction(cls, method, shape, dtype):
    x = torch.arange(1, math.prod(shape) + 1, dtype=dtype).reshape(shape)
    if x.ndim == 2:
        x = x.T
    projector = cls(sparse_ratio=0.5, sparse_type=method)
    small = projector.project(x, 0)
    if cls is TensorGradUnstructuredProjector:
        indices = projector._indices
        assert indices.numel() == math.ceil(x.numel() / 2)
        assert indices.unique().numel() == indices.numel()
        expected = torch.zeros_like(x).reshape(-1)
        expected[indices] = x.reshape(-1)[indices]
        expected = expected.reshape(x.shape)
    else:
        expected = x.clone()
        for mode, mask in enumerate(projector.masks):
            assert mask.sum() == math.ceil((0.5 ** (1 / sum(d != 1 for d in x.shape)) if x.shape[mode] != 1 else 1) * x.shape[mode])
            index = [slice(None)] * x.ndim
            index[mode] = ~mask
            expected[tuple(index)] = 0
        for indices, size in zip(projector.indices, x.shape):
            assert indices.unique().numel() == indices.numel()
            assert ((0 <= indices) & (indices < size)).all()
    torch.testing.assert_close(projector.project_back(small), expected)
    buffer = torch.full_like(x, 7)
    returned = projector.project_back(small, buffer, alpha=-2, accumulate=False)
    assert returned is buffer
    torch.testing.assert_close(buffer, -2 * expected)
    projector.project_back(small, buffer, alpha=3, accumulate=True)
    torch.testing.assert_close(buffer, expected)


def test_integer_boundary_and_zero_probability_scores():
    projector = TensorGradSparseProjector([0.5, 1.0], "randk")
    small = projector.project(torch.zeros(4, 3), 0)
    assert small.shape == (2, 3)
    for cls in [TensorGradSparseProjector, TensorGradUnstructuredProjector]:
        projector = cls(0.5, "probability")
        assert torch.isfinite(projector.project(torch.zeros(4, 5), 0)).all()


@pytest.mark.parametrize("mode", sorted(UPDATE_MODES))
def test_update_schedules_resume_and_validation(mode):
    first = UpdateGapScheduler(2, 8, mode, total_iters=20)
    updates = [i for i in range(11) if first.should_update(i)]
    second = UpdateGapScheduler(99, 99)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    assert [first.should_update(i) for i in range(11, 40)] == [second.should_update(i) for i in range(11, 40)]
    assert updates[0] == 0
    assert 2 <= first.compute_gap(1000) <= 8
    with pytest.raises(ValueError):
        UpdateGapScheduler(0, 1)
    with pytest.raises(ValueError):
        UpdateGapScheduler(1, 1, total_iters=0)


def test_fixed_updates_and_instance_independence():
    first, second = GaLoreProjector(1), GaLoreProjector(1)
    assert first.update_gap_scheduler is not second.update_gap_scheduler
    updates = []
    for i in range(201):
        first.project(torch.eye(3), i)
        if first.should_update:
            updates.append(i)
    assert updates == [0, 100, 200]
    second.project(torch.eye(3), 0)
    assert second.should_update


@pytest.mark.parametrize("side", ["left", "right", "full"])
@pytest.mark.parametrize("svd", ["truncated_svd", "full_svd", "randomized_svd"])
def test_galore_back_buffers_and_portable_dtype(side, svd):
    x = torch.arange(1, 21, dtype=torch.float32).reshape(4, 5)
    projector = GaLoreProjector(2, svd_type=svd, galore_2d_proj_type=side)
    compressed = projector.project(x, 0)
    expected = projector.project_back(compressed)
    buffer = torch.ones_like(x)
    torch.testing.assert_close(projector.project_back(compressed, buffer, -2, True), 1 - 2 * expected)
    torch.testing.assert_close(projector.project_back(compressed, buffer, 3, False), 3 * expected)
    bases = projector.ortho_matrix if isinstance(projector.ortho_matrix, list) else [projector.ortho_matrix]
    assert all(b.untyped_storage().nbytes() == b.numel() * b.element_size() for b in bases)
    target = torch.nn.Parameter(x.double())
    restored = projector_from_state(copy.deepcopy(projector.state_dict()), target)
    restored_small = restored.project(x.double(), 1)
    assert restored.project_back(restored_small).dtype == torch.float64


def test_tucker_projector_buffers_and_global_backend():
    x = torch.arange(1, 25, dtype=torch.float64).reshape(2, 3, 4)
    before = tl.get_backend()
    projector = TensorGradLowRankProjector([2, 2, 2], n_iter_max=2)
    small = projector.project(x, 0)
    expected = projector.project_back(small)
    buffer = torch.ones_like(x)
    torch.testing.assert_close(projector.project_back(small, buffer, -2, True), 1 - 2 * expected)
    torch.testing.assert_close(projector.project_back(small, buffer, 2, False), 2 * expected)
    assert tl.get_backend() == before


@pytest.mark.parametrize("preset", [ParallelTG, ULTG])
@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
@pytest.mark.parametrize("svd", ["truncated_svd", "randomized_svd", "full_svd"])
def test_prepared_public_first_two_steps(preset, shape, svd):
    model = Weights(shape)
    optimizer, scheduler = preset(model, svd, 2, n_iter_max_tucker=2)
    for _ in range(2):
        model.weight.grad = model.weight.detach().clone()
        original_grad = model.weight.grad.clone()
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()
        scheduler.step()
        torch.testing.assert_close(model.weight.grad, original_grad)
        assert torch.isfinite(model.weight).all()
    assert "first_proj" in optimizer.state[model.weight]


@pytest.mark.parametrize("algorithm", ["sgd", "adamw", "tensorgrad", "tensorgrad_sum"])
def test_algorithm_configuration_and_parameter_grouping(algorithm):
    model = Weights()
    config = TensorGRaDConfig(DataConfig(), OptimizerConfig(optimizer_type=algorithm, rank=2, exclude_first_parameter=True))
    optimizer, _ = setup_optimizer_and_scheduler(config, model)
    expected = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW,
                "tensorgrad": TensorGRaD, "tensorgrad_sum": TensorGRaD}[algorithm]
    assert type(optimizer) is expected
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
    assert len({id(p) for g in optimizer.param_groups for p in g["params"]}) == len(list(model.parameters()))
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    if algorithm.startswith("tensorgrad"):
        assert "first_proj" not in optimizer.state[model.weight]


@pytest.mark.parametrize("betas", [(0, 0), (0.9, 0.999)])
@pytest.mark.parametrize("eps", [0, 1e-8, 1e-2])
@pytest.mark.parametrize("decay", [0, 0.1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_unprojected_adamw_parity(betas, eps, decay, dtype):
    first = torch.nn.Parameter(torch.tensor([1, -2], dtype=dtype))
    second = torch.nn.Parameter(first.detach().clone())
    actual = TensorGRaD([first], lr=0.1, betas=betas, eps=eps, weight_decay=decay)
    reference = torch.optim.AdamW([second], lr=0.1, betas=betas, eps=eps, weight_decay=decay)
    for value in [1e-9, -0.3, 0.1, 2.0]:
        gradient = torch.tensor([value, value * 2], dtype=dtype)
        if dtype.is_complex:
            gradient += 3j * value
        first.grad, second.grad = gradient.clone(), gradient.clone()
        actual.step()
        reference.step()
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_closure_gradient_context_and_exception():
    p = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = TensorGRaD([p])
    def closure():
        optimizer.zero_grad()
        loss = p**2
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    assert loss.item() == 4
    value, state = p.clone(), copy.deepcopy(optimizer.state_dict())
    def failure():
        raise RuntimeError("closure failed")
    with pytest.raises(RuntimeError, match="closure failed"):
        optimizer.step(failure)
    torch.testing.assert_close(value, p)
    assert optimizer.state_dict()["state"][0]["step"] == state["state"][0]["step"]


def test_sequential_residual_and_gradient_ownership():
    p = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float64))
    group = dict(params=[p], rank=1, proj_type="low_rank", second_proj_type="unstructured_sparse",
                 sparse_ratio=0.5, second_sparse_ratio=0.5, lambda_sparse=1.0)
    optimizer = TensorGRaD([group], eps=0.1, matrix_only=False)
    p.grad = torch.arange(1, 13, dtype=p.dtype).reshape(p.shape)
    source = p.grad.clone()
    optimizer.step()
    state = optimizer.state[p]
    first, second = state["first_proj"], state["second_proj"]
    # Read the actual second branch's moment: at first step it is (1-beta1)*input.
    expected = source - first.project_back(first.project(source, 0))
    expected = expected.reshape(-1)[second._indices]
    torch.testing.assert_close(state["second_exp_avg"], expected * 0.1)
    torch.testing.assert_close(p.grad, source)


@pytest.mark.parametrize("cls", [TensorGradSparseProjector, TensorGradUnstructuredProjector])
def test_sparse_moments_reset_bias_clock(cls):
    p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float64))
    kind = "structured_sparse" if cls is TensorGradSparseProjector else "unstructured_sparse"
    optimizer = TensorGRaD([dict(params=[p], rank=1, proj_type=kind, projection_mode="single",
        sparse_ratio=0.25, update_proj_gap=1, update_proj_gap_end=1, lambda_sparse=1)], betas=(0.5, 0.5))
    for i in range(2):
        p.grad = torch.arange(1, 17, dtype=p.dtype).reshape(4, 4).roll(i, 0)
        optimizer.step()
        state = optimizer.state[p]
        projector = state["first_proj"]
        selected = projector._transform(p.grad) if cls is TensorGradSparseProjector else p.grad.reshape(-1)[projector._indices]
        torch.testing.assert_close(state["first_exp_avg"], selected * 0.5)
        torch.testing.assert_close(state["first_exp_avg_sq"], selected.square() * 0.5)
        assert state["first_step"] == 1


def test_uniform_scaling_analytic_expectations_and_topk_guard():
    gradient = torch.arange(1, 5, dtype=torch.float64)
    masks = list(combinations(range(4), 2))
    for scaling in ["none", "energy", "unbiased"]:
        projector = TensorGradUnstructuredProjector(0.5, "randk", scaling=scaling)
        projector.project(gradient.reshape(2, 2), 0)
        values = []
        for mask in masks:
            projector._indices = torch.tensor(mask)
            values.append(projector.project_back(gradient[list(mask)]).reshape(-1))
        expected_mean = gradient * {"none": 0.5, "energy": math.sqrt(0.5), "unbiased": 1}[scaling]
        torch.testing.assert_close(torch.stack(values).mean(0), expected_mean)
        if scaling == "energy":
            torch.testing.assert_close(torch.stack(values).square().sum(1).mean(), gradient.square().sum())
    with pytest.raises(ValueError, match="uniform"):
        TensorGradUnstructuredProjector(0.5, "topk", scaling="unbiased")


@pytest.mark.parametrize("first", ["low_rank", "structured_sparse", "unstructured_sparse"])
@pytest.mark.parametrize("second", ["low_rank", "structured_sparse", "unstructured_sparse"])
@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
def test_composite_checkpoint_trajectory_and_dtype(first, second, shape):
    initial = Weights(shape)
    group = dict(rank=2, second_rank=2, proj_type=first, second_proj_type=second,
        sparse_type="randk", second_sparse_type="randk", update_proj_gap=2,
        update_proj_gap_end=2, sparse_ratio=0.5, second_sparse_ratio=0.5,
        n_iter_max_tucker=2, random_state=13, lambda_sparse=0.4)
    actual = TensorGRaD([dict(group, params=[initial.weight]), {"params": [initial.bias]}],
                       matrix_only=False, use_sum=False, support_complex=True)
    for i in range(2):
        initial.weight.grad = torch.sin(initial.weight.detach() + i)
        initial.bias.grad = torch.ones_like(initial.bias)
        actual.step()
    stream = io.BytesIO()
    torch.save(actual.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored_model = copy.deepcopy(initial)
    restored = TensorGRaD([dict(group, params=[restored_model.weight]), {"params": [restored_model.bias]}],
                         matrix_only=True, use_sum=True, support_complex=False)
    restored.load_state_dict(saved)
    assert restored.use_sum is False and restored.matrix_only is False and restored.support_complex is True
    for i in range(2, 6):
        for model, optimizer in [(initial, actual), (restored_model, restored)]:
            model.weight.grad = torch.sin(model.weight.detach() + i)
            model.bias.grad = torch.ones_like(model.bias)
            optimizer.step()
        torch.testing.assert_close(initial.weight, restored_model.weight, rtol=1e-12, atol=1e-12)
    target = Weights(shape, torch.float32)
    cast_optimizer = TensorGRaD([dict(group, params=[target.weight]), {"params": [target.bias]}], matrix_only=False)
    cast_optimizer.load_state_dict(saved)
    target.weight.grad = torch.ones_like(target.weight)
    target.bias.grad = torch.ones_like(target.bias)
    cast_optimizer.step()
    assert target.weight.dtype == torch.float32


@pytest.mark.parametrize("name", sorted(SCHEDULER_NAMES))
def test_learning_rate_schedule_contract(name):
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = TensorGRaD([p])
    scheduler = get_scheduler(name, optimizer, T_max=3, step_size=2)
    optimizer.step()
    scheduler.step(1.0) if name == "ReduceLROnPlateau" else scheduler.step()
    state = copy.deepcopy(scheduler.state_dict())
    scheduler.load_state_dict(state)
    assert scheduler.state_dict() == state


@pytest.mark.parametrize("epoch", [None, 0, 3])
@pytest.mark.parametrize("components", [tuple(name for name, included in zip(("model", "optimizer", "scheduler", "regularizer"), flags) if included) for flags in product((False, True), repeat=4)])
def test_training_state_optional_roundtrip(tmp_path, epoch, components):
    model = torch.nn.Linear(2, 1)
    optimizer = TensorGRaD(model.parameters())
    scheduler = get_scheduler("cosine", optimizer)
    values = dict(model=model, optimizer=optimizer, scheduler=scheduler, regularizer=torch.nn.Linear(2, 1))
    selected = {name: values[name] for name in components}
    save_training_state(tmp_path, "small", epoch=epoch, **selected)
    before = model.weight.detach().clone()
    if "model" in selected:
        with torch.no_grad():
            model.weight.add_(9)
    result = load_training_state(tmp_path, "small", **selected)
    assert result.epoch is epoch or result.epoch == epoch
    if "model" in selected:
        torch.testing.assert_close(model.weight, before)


def test_training_checkpoint_rng_scheduler_and_rollback(tmp_path):
    model = Weights()
    optimizer = TensorGRaD([dict(params=[model.weight], rank=2, sparse_type="randk",
        second_sparse_type="randk", update_proj_gap=1, update_proj_gap_end=1)], matrix_only=False)
    scheduler = get_scheduler("cosine", optimizer, T_max=10)
    model.weight.grad = model.weight.detach().sin()
    optimizer.step()
    scheduler.step()
    save_training_state(tmp_path, "model", model, optimizer, scheduler)
    draws = (random.random(), np.random.rand(), torch.rand(2))
    model.weight.grad = model.weight.detach().cos()
    optimizer.step()
    scheduler.step()
    expected = model.weight.detach().clone()
    load_training_state(tmp_path, "model", model, optimizer, scheduler)
    assert random.random() == draws[0]
    assert np.random.rand() == draws[1]
    torch.testing.assert_close(torch.rand(2), draws[2], rtol=0, atol=0)
    model.weight.grad = model.weight.detach().cos()
    optimizer.step()
    scheduler.step()
    torch.testing.assert_close(model.weight, expected)
    before = model.weight.detach().clone()
    manifest = torch.load(tmp_path / "manifest.pt", weights_only=True)
    manifest["version"] = 99
    torch.save(manifest, tmp_path / "manifest.pt")
    with pytest.raises(ValueError, match="version"):
        load_training_state(tmp_path, "model", model, optimizer, scheduler)
    torch.testing.assert_close(before, model.weight)
    manifest["version"] = 1
    manifest.pop("epoch")
    torch.save(manifest, tmp_path / "manifest.pt")
    with pytest.raises(ValueError, match="incomplete"):
        load_training_state(tmp_path, "model", model, optimizer, scheduler)
    torch.testing.assert_close(before, model.weight)


def test_invalid_config_fails_before_state_changes():
    model = Weights()
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="unknown preset"):
        ParallelTG(model, "truncated_svd", 2, typo=1)
    with pytest.raises(ValueError):
        OptimizerConfig(sparse_type="unknown")
    optimizer = TensorGRaD([dict(params=[model.weight], rank=2)])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.param_groups[0]["update_proj_gap"] = 0
    with pytest.raises(ValueError):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(before, model.weight)
    optimizer.param_groups[0]["update_proj_gap"] = 2
    optimizer.step()
    assert optimizer.state


def test_portable_state_invalid_version_does_not_mutate():
    p = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = TensorGRaD([dict(params=[p], rank=1)])
    p.grad = torch.ones_like(p)
    optimizer.step()
    saved = optimizer.state_dict()
    bad = copy.deepcopy(saved)
    bad["tensorgrad"]["version"] = 99
    with pytest.raises(ValueError, match="version"):
        optimizer.load_state_dict(bad)
    assert optimizer.state[p]["step"] == 1
    bad = copy.deepcopy(saved)
    bad["tensorgrad"]["projectors"][0]["first_proj"]["data"] = {}
    with pytest.raises(ValueError, match="incomplete"):
        optimizer.load_state_dict(bad)
    assert optimizer.state[p]["step"] == 1




@pytest.mark.parametrize("preset", [ParallelTG, ULTG])
def test_linear_33_public_defaults_and_callable(preset):
    model = torch.nn.Linear(33, 33, bias=False)
    optimizer, scheduler = preset(model, tl.truncated_svd, 2)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 33)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    stream.seek(0)
    torch.load(stream, weights_only=True)


@pytest.mark.parametrize("kind", ["low_rank", "structured_sparse", "unstructured_sparse"])
@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
def test_single_projector_configuration(kind, shape):
    model = Weights(shape)
    optimizer = TensorGRaD([dict(params=[model.weight], rank=2, proj_type=kind, projection_mode="single",
        n_iter_max_tucker=2)], matrix_only=False)
    for _ in range(2):
        model.weight.grad = model.weight.detach().sin()
        optimizer.step()
    assert "second_proj" not in optimizer.state[model.weight]


@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
def test_float32_to_float64_state_and_complex_projection(shape):
    model = Weights(shape, torch.float32)
    config = dict(rank=2, n_iter_max_tucker=2, update_proj_gap=2, update_proj_gap_end=2)
    optimizer = TensorGRaD([dict(config, params=[model.weight])], matrix_only=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    target = Weights(shape, torch.float64)
    restored = TensorGRaD([dict(config, params=[target.weight])], matrix_only=False)
    restored.load_state_dict(optimizer.state_dict())
    target.weight.grad = torch.ones_like(target.weight)
    restored.step()
    assert restored.state[target.weight]["first_exp_avg"].dtype == torch.float64
    complex_model = Weights(shape, torch.complex128)
    complex_optimizer = TensorGRaD([dict(config, params=[complex_model.weight])], matrix_only=False, support_complex=True)
    for _ in range(2):
        complex_model.weight.grad = complex_model.weight.detach() * (1 + 1j)
        complex_optimizer.step()
    assert torch.isfinite(complex_model.weight).all()


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("problem", ["missing_square", "bad_shape", "missing_clock", "negative_clock", "bad_indices", "bad_basis"])
def test_invalid_compressed_moments_rejected_atomically(problem):
    p = torch.nn.Parameter(torch.ones(4, 5))
    optimizer = TensorGRaD([dict(params=[p], rank=2, sparse_type="randk", second_sparse_type="randk")])
    p.grad = torch.ones_like(p)
    optimizer.step()
    before = optimizer.state_dict()
    bad = copy.deepcopy(before)
    state = bad["state"][0]
    if problem == "missing_square":
        state.pop("first_exp_avg_sq")
    elif problem == "bad_shape":
        state["first_exp_avg_sq"] = torch.zeros(999)
    elif problem == "missing_clock":
        state.pop("second_step")
    elif problem == "negative_clock":
        state["first_step"] = -1
    elif problem == "bad_indices":
        bad["tensorgrad"]["projectors"][0]["second_proj"]["data"]["_indices"][0] = -1
    else:
        bad["tensorgrad"]["projectors"][0]["first_proj"]["data"]["ortho_matrix"] = torch.ones(999, 1)
    bad["tensorgrad"]["method"]["use_sum"] = True
    bad["param_groups"][0]["lr"] = 123
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert_nested_equal(optimizer.state_dict(), before)


def test_training_load_component_failure_rolls_back_every_component(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = TensorGRaD(model.parameters())
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    save_training_state(tmp_path, "m", model, optimizer)
    manifest = torch.load(tmp_path / "manifest.pt", weights_only=True)
    path = tmp_path / manifest["components"]["optimizer"]
    bad = torch.load(path, weights_only=True)
    bad["state"][0].pop("exp_avg_sq")
    torch.save(bad, path)
    with torch.no_grad():
        model.weight.add_(2)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = optimizer.state_dict()
    with pytest.raises(ValueError, match="moment"):
        load_training_state(tmp_path, "m", model, optimizer)
    assert_nested_equal(model.state_dict(), before_model)
    assert_nested_equal(optimizer.state_dict(), before_optimizer)


def test_failed_save_keeps_previous_generation_loadable(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("tdecomp.grad_proj.tensorgrad.training_utils.training_state")
    model = torch.nn.Linear(2, 1)
    optimizer = TensorGRaD(model.parameters())
    save_training_state(tmp_path, "m", model, optimizer, epoch=3)
    original = model.weight.detach().clone()
    old_manifest = torch.load(tmp_path / "manifest.pt", weights_only=True)
    with torch.no_grad():
        model.weight.add_(2)
    real_save = module._atomic_save
    def fail_on_optimizer(value, path):
        if path.name.endswith("_optimizer.pt"):
            raise OSError("injected write failure")
        real_save(value, path)
    monkeypatch.setattr(module, "_atomic_save", fail_on_optimizer)
    with pytest.raises(OSError, match="injected"):
        save_training_state(tmp_path, "m", model, optimizer, epoch=4)
    assert_nested_equal(torch.load(tmp_path / "manifest.pt", weights_only=True), old_manifest)
    result = load_training_state(tmp_path, "m", model, optimizer)
    assert result.epoch == 3
    torch.testing.assert_close(model.weight, original)


def test_random_projectors_do_not_change_application_rng():
    before_torch = torch.random.get_rng_state()
    before_numpy = np.random.get_state()
    for cls in [TensorGradSparseProjector, TensorGradUnstructuredProjector]:
        projector = cls(0.5, "randk", random_state=42)
        projector.project(torch.ones(3, 4), 0)
    projector = GaLoreProjector(1, svd_type="randomized_svd", random_state=42)
    projector.project(torch.ones(8, 9), 0)
    projector = TensorGradLowRankProjector(2, svd_type="randomized_svd", random_state=42, n_iter_max=2)
    projector.project(torch.ones(3, 4, 5), 0)
    torch.testing.assert_close(torch.random.get_rng_state(), before_torch, rtol=0, atol=0)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]


@pytest.mark.parametrize("side", ["left", "right", "std", "reverse_std"])
def test_galore_sparse_public_modes(side):
    x = torch.arange(1, 21, dtype=torch.float64).reshape(4, 5)
    projector = GaLoreSparseProjector(0.5, "randk", proj_type=side)
    small = projector.project(x, 0)
    recovered = projector.project_back(small)
    assert torch.equal(recovered[recovered != 0], x[recovered != 0])
    buffer = torch.ones_like(x)
    torch.testing.assert_close(projector.project_back(small, buffer, -1, True), 1 - recovered)


def test_activation_checkpoint_option_invokes_matmul_path(monkeypatch):
    import importlib
    module = importlib.import_module("tdecomp.grad_proj.tensorgrad.projectors.galore_projector")
    calls = []
    original = module.optional_checkpoint_matmul
    def recording(a, b, activation_checkpoint):
        calls.append(activation_checkpoint)
        return original(a, b, activation_checkpoint)
    monkeypatch.setattr(module, "optional_checkpoint_matmul", recording)
    projector = GaLoreProjector(1, activation_checkpoint=True)
    compressed = projector.project(torch.ones(3, 4), 0)
    projector.project_back(compressed)
    assert calls == [True, True]


def test_custom_callable_works_and_portable_save_is_explicit():
    def custom(matrix):
        return torch.linalg.svd(matrix, full_matrices=False)
    model = Weights()
    optimizer, _ = ParallelTG(model, custom, 2)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    with pytest.raises(ValueError, match="custom SVD"):
        optimizer.state_dict()




@pytest.mark.parametrize("sum_mode", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_two_known_linear_projectors_independent_reference(monkeypatch, sum_mode, reverse):
    import importlib
    module = importlib.import_module("tdecomp.grad_proj.tensorgrad.tensorgrad")
    from tdecomp.grad_proj.tensorgrad.projectors._common import write_back
    class LinearProjector:
        def __init__(self, mask):
            self.mask, self.should_update, self.received = mask, False, []
        def _check_input(self, x):
            pass
        def project(self, x, iteration):
            self.received.append(x.clone())
            self.should_update = iteration == 0
            return self.mask * x
        def project_back(self, compressed, output_buffer=None, alpha=1, accumulate=False):
            return write_back(self.mask * compressed, output_buffer, alpha, accumulate)
    masks = [torch.tensor([[1., 1.], [0., 0.]], dtype=torch.float64),
             torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)]
    if reverse:
        masks.reverse()
    first, second = (LinearProjector(mask) for mask in masks)
    monkeypatch.setattr(module, "get_projector", lambda *args, **kwargs: (first, second))
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    optimizer = TensorGRaD([dict(params=[p], rank=1)], use_sum=sum_mode, betas=(0, 0), eps=0.1, lr=0.2)
    gradient = torch.tensor([[1., 2.], [3., 4.]], dtype=p.dtype)
    p.grad = gradient.clone()
    optimizer.step()
    first_gradient = gradient * masks[0]
    second_input = gradient if sum_mode else gradient - first_gradient
    second_gradient = second_input * masks[1]
    expected_update = first_gradient / (first_gradient.abs() + 0.1) * masks[0]
    expected_update += second_gradient / (second_gradient.abs() + 0.1) * masks[1]
    torch.testing.assert_close(p, 1 - 0.2 * expected_update)
    torch.testing.assert_close(second.received[-1], second_input)
    torch.testing.assert_close(p.grad, gradient)


def test_changed_sign_and_rotated_basis_reset_both_moments(monkeypatch):
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    optimizer = TensorGRaD([dict(params=[p], rank=2, projection_mode="single",
        update_proj_gap=1, update_proj_gap_end=1)], betas=(0.5, 0.75))
    gradient = torch.tensor([[1., 2.], [3., 4.]], dtype=p.dtype)
    bases = [torch.eye(2, dtype=p.dtype), -torch.eye(2, dtype=p.dtype),
             torch.tensor([[0., -1.], [1., 0.]], dtype=p.dtype)]
    iteration = iter(bases)
    monkeypatch.setattr(GaLoreProjector, "get_orthogonal_matrix", lambda *args, **kwargs: next(iteration))
    for basis in bases:
        p.grad = gradient.clone()
        optimizer.step()
        state = optimizer.state[p]
        compressed = basis.mH @ gradient
        torch.testing.assert_close(state["first_exp_avg"], compressed * 0.5)
        torch.testing.assert_close(state["first_exp_avg_sq"], compressed.square() * 0.25)
        assert state["first_step"] == 1


def test_changed_index_order_resets_moments(monkeypatch):
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    optimizer = TensorGRaD([dict(params=[p], rank=1, projection_mode="single",
        proj_type="unstructured_sparse", sparse_ratio=0.5, update_proj_gap=1, update_proj_gap_end=1)],
        betas=(0.5, 0.75))
    gradient = torch.tensor([[1., 2.], [3., 4.]], dtype=p.dtype)
    indices = [torch.tensor([0, 3]), torch.tensor([3, 0])]
    iteration = iter(indices)
    def build(projector, x):
        projector._indices = next(iteration)
    monkeypatch.setattr(TensorGradUnstructuredProjector, "_build_indices", build)
    for index in indices:
        p.grad = gradient.clone()
        optimizer.step()
        state = optimizer.state[p]
        compressed = gradient.reshape(-1)[index]
        torch.testing.assert_close(state["first_exp_avg"], compressed * 0.5)
        torch.testing.assert_close(state["first_exp_avg_sq"], compressed.square() * 0.25)
        assert state["first_step"] == 1


def test_optional_modules_import_without_distributed_or_cuda_feature():
    import importlib
    importlib.import_module("tdecomp.grad_proj.tensorgrad.mem_trace")
    module = importlib.import_module("tdecomp.grad_proj.tensorgrad.training_utils.training_state_async")
    try:
        module._distributed_api()
    except ImportError as exc:
        assert "distributed" in str(exc)




@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_mutated_branch_weight_is_rejected_before_update(value):
    p = torch.nn.Parameter(torch.ones(3, 4))
    optimizer = TensorGRaD([dict(params=[p], rank=1)])
    p.grad = torch.ones_like(p)
    optimizer.step()
    before_parameter = p.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict()["state"])
    optimizer.param_groups[0]["lambda_sparse"] = value
    with pytest.raises(ValueError, match="lambda_sparse"):
        optimizer.step()
    torch.testing.assert_close(p, before_parameter)
    assert_nested_equal(optimizer.state_dict()["state"], before_state)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 3, 4)])
def test_empty_sparse_input_is_rejected_before_schedule(shape):
    for cls in [TensorGradSparseProjector, TensorGradUnstructuredProjector]:
        projector = cls()
        with pytest.raises(ValueError, match="nonempty"):
            projector.project(torch.empty(shape), 0)
        assert projector.update_gap_scheduler.last_iter == -1


def test_false_reset_policy_and_nonfinite_input_reject_early():
    model = Weights()
    with pytest.raises(ValueError, match="reset_sparse"):
        ParallelTG(model, "truncated_svd", 2, reset_sparse_optimizer_states=False)
    projector = GaLoreProjector(1)
    with pytest.raises(ValueError, match="finite"):
        projector.project(torch.full((3, 4), float("nan")), 0)
    assert projector.update_gap_scheduler.last_iter == -1




def test_custom_tensor_decomposer_computes_but_portable_state_rejects_it():
    class CoordinateDecomposer:
        def __init__(self, **kwargs):
            pass

        def decompose(self, tensor, **kwargs):
            factors = [torch.eye(size, dtype=tensor.dtype, device=tensor.device)[:, :1]
                       for size in tensor.shape]
            return tensor[:1, :1, :1], factors

    x = torch.arange(1, 61, dtype=torch.float64).reshape(3, 4, 5)
    projector = TensorGradLowRankProjector(1, tensor_decomposer_type=CoordinateDecomposer)
    compressed = projector.project(x, 0)
    torch.testing.assert_close(compressed, x[:1, :1, :1])
    expected = torch.zeros_like(x)
    expected[0, 0, 0] = x[0, 0, 0]
    torch.testing.assert_close(projector.project_back(compressed), expected)
    with pytest.raises(ValueError, match="custom tensor decomposer"):
        projector.state_dict()
