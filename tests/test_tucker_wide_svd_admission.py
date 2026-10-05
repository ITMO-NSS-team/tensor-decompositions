"""Actual real-layer spatial unfolding memory contract and tall left completion."""
import pytest
import torch

from experiments.hypotheses import synthetic_tucker_common as common
from experiments.hypotheses.run_h06_synthetic import st_hosvd


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_actual_spatial_unfolding_never_requests_quadratic_right_basis(monkeypatch):
    original = torch.linalg.svd
    calls = []
    def guarded(matrix, *, full_matrices):
        calls.append((tuple(matrix.shape), full_matrices))
        assert not full_matrices if matrix.shape[1] >= matrix.shape[0] else full_matrices
        return original(matrix, full_matrices=full_matrices)
    monkeypatch.setattr(torch.linalg, "svd", guarded)
    weight = torch.randn(256, 256, 3, 3, generator=torch.Generator().manual_seed(28))
    core, factors, _ = st_hosvd(weight, (64, 64, 3, 3), order=(2, 3, 0, 1))
    assert calls[0] == ((3, 196608), False)
    assert tuple(core.shape) == (64, 64, 3, 3)
    assert [tuple(f.shape) for f in factors] == [(256, 64), (256, 64), (3, 3), (3, 3)]
    assert torch.isfinite(core).all()


def test_tall_full_rank_keeps_complete_left_null_space_and_declared_core_shape():
    weight = torch.randn(12, 1, 1, 1, generator=torch.Generator().manual_seed(38), dtype=torch.float64)
    core, factors = common.exact_hosvd(weight, (12, 1, 1, 1))
    assert tuple(core.shape) == (12, 1, 1, 1)
    assert factors[0].shape == (12, 12)
    torch.testing.assert_close(common.reconstruct(core, factors), weight, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(factors[0].T @ factors[0], torch.eye(12, dtype=weight.dtype), rtol=1e-12, atol=1e-12)
