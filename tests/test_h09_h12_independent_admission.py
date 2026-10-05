"""Independent optimizer-coordinate and Tucker core-tangent checks."""
import pytest
import tensorly as tl
import torch

from experiments.hypotheses import run_h09_synthetic as h09
from experiments.hypotheses import run_h12_rgn as h12


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_compressed_adam_step_matches_independent_torch_adam_in_frozen_coordinates():
    gen = torch.Generator().manual_seed(305)
    parameter = torch.nn.Parameter(torch.randn(12, 7, generator=gen))
    compressed = h09.CompressedAdam(parameter.shape, "cpu", rank=3, refresh="fixed", seed=15)
    virtual = torch.nn.Parameter(torch.zeros(3, 7))
    optimizer = torch.optim.AdamW([virtual], lr=.001, weight_decay=0)
    feedback = torch.zeros_like(parameter)
    for step in range(3):
        parameter.grad = torch.randn(parameter.shape, generator=gen)
        before = parameter.detach().clone()
        gradient_before = parameter.grad.clone()
        h = gradient_before + feedback
        compressed.step(parameter, step)
        g = compressed.q.T @ h
        virtual_before = virtual.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        virtual.grad = g
        optimizer.step()
        expected_delta = compressed.q @ (virtual.detach() - virtual_before)
        # Compare updated parameters: subtracting two O(1) FP32 weights to
        # isolate an O(1e-3) step introduces avoidable cancellation.
        torch.testing.assert_close(parameter, before + expected_delta, rtol=1e-6, atol=1e-7)
        feedback = h - compressed.q @ g
        torch.testing.assert_close(compressed.e, feedback, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(parameter.grad, gradient_before, rtol=0, atol=0)


def test_refresh_resets_both_moments_local_bias_and_preserves_full_coordinate_feedback():
    gen = torch.Generator().manual_seed(731)
    parameter = torch.nn.Parameter(torch.randn(12, 7, generator=gen))
    compressed = h09.CompressedAdam(parameter.shape, "cpu", rank=3, refresh="fixed", seed=17)
    parameter.grad = torch.randn(parameter.shape, generator=gen)
    compressed.step(parameter, 0)
    feedback_before = compressed.e.clone()
    parameter.grad = torch.randn(parameter.shape, generator=gen)
    h = parameter.grad + feedback_before
    compressed.step(parameter, 40)
    expected = compressed.q.T @ h
    assert compressed.tau == 1 and compressed.updates == 2
    torch.testing.assert_close(compressed.m, .1 * expected)
    torch.testing.assert_close(compressed.v, .001 * expected.square())
    torch.testing.assert_close(compressed.e, h - compressed.q @ expected)


def test_tucker_core_tangent_is_full_rank_and_orthonormal_with_noncontiguous_core():
    gen = torch.Generator().manual_seed(40012)
    with tl.backend_context("pytorch"):
        target = torch.randn(5, 4, 3, generator=gen, dtype=torch.float64)
        core, factors = h12.hosvd(target, (2, 2, 2))
        # Force a non-contiguous tensor even if hosvd returns contiguous data.
        core = core.transpose(0, 1)
        matrix, coordinates, complements = h12.tangent_matrix(core, factors)
        block = matrix[:, :core.numel()]
        assert int(torch.linalg.matrix_rank(block)) == core.numel()
        torch.testing.assert_close(block.T @ block, torch.eye(core.numel(), dtype=core.dtype), atol=1e-12, rtol=1e-12)
        delta_core = torch.randn(core.shape, generator=gen, dtype=core.dtype)
        tangent_vector = torch.zeros(matrix.shape[1], dtype=core.dtype)
        tangent_vector[:core.numel()] = delta_core.reshape(-1)
        independent_delta = torch.einsum("abc,ia,jb,kc->ijk", delta_core, *factors).reshape(-1)
        torch.testing.assert_close(matrix @ tangent_vector, independent_delta, atol=1e-12, rtol=1e-12)
        changed = h12.coordinate_perturbation(core, factors, coordinates, complements, tangent_vector, 1e-5)
        initial = h12.reconstruct(core, factors)
        torch.testing.assert_close((changed - initial).reshape(-1) / 1e-5, independent_delta, atol=1e-9, rtol=1e-9)
