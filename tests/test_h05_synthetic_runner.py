import math

import pytest
import tensorly as tl
import torch

from experiments.hypotheses import run_h05_synthetic as h05
from experiments.hypotheses import synthetic_tucker_common as common


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_independent_tensor_als_and_fft_admission():
    results = h05.admission()
    assert results["tensorsketch_explicit_fft_collisions"] == "passed"
    assert results["unfolding_kronecker_order"] == "passed"
    assert results["exact_LS_vs_independent_gelsd"] == "passed"
    assert results["qr_core_transport"] == "passed"


def test_same_composite_sketch_applies_to_operator_and_rhs():
    gen = torch.Generator().manual_seed(67)
    sketch = h05.TensorSketch.draw((2, 3, 4), 7, 12, dtype=torch.float64)
    a = torch.randn(24, 3, generator=gen, dtype=torch.float64)
    b = torch.randn(24, 5, generator=gen, dtype=torch.float64)
    explicit = sketch.explicit()
    torch.testing.assert_close(sketch.apply(a), explicit @ a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sketch.apply(b), explicit @ b, rtol=1e-12, atol=1e-12)
    assert torch.all(explicit.square().sum(0) == 1)


def test_adversarial_collision_small_sketch_residual_cannot_hide_true_error():
    sketch = h05.TensorSketch((2, 2), 1, (torch.zeros(2, dtype=torch.long),) * 2,
                              (torch.ones(2, dtype=torch.float64),) * 2)
    a = torch.tensor([[1.], [0.], [0.], [0.]], dtype=torch.float64)
    b = torch.tensor([[0.], [1.], [0.], [0.]], dtype=torch.float64)
    z = h05.svd_ls(sketch.apply(a), sketch.apply(b))
    assert float((sketch.apply(a) @ z - sketch.apply(b)).norm()) == 0
    assert float((a @ z - b).norm()) == pytest.approx(math.sqrt(2))
    with pytest.raises(ArithmeticError, match="true LS residual"):
        h05.local_guard(a, b, z, h05.svd_ls(a, b))


def test_svd_rcond_is_identical_and_no_hidden_ridge_in_rank_deficient_case():
    a = torch.diag(torch.tensor([1., 1e-8, 0.], dtype=torch.float64))
    b = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(h05.svd_ls(a, b), torch.diag(torch.tensor([1., 0., 0.], dtype=torch.float64)))


def test_qr_transport_preserves_nonorthogonal_tucker_reconstruction():
    gen = torch.Generator().manual_seed(561)
    core = torch.randn(2, 2, 2, generator=gen, dtype=torch.float64)
    factors = [torch.randn(n, 2, generator=gen, dtype=torch.float64) for n in (3, 4, 5)]
    original = common.reconstruct(core, factors)
    updated_core, updated_factors, error = h05.factor_qr(core, factors, 1, factors[1])
    assert error < 1e-12
    torch.testing.assert_close(common.reconstruct(updated_core, updated_factors), original, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(updated_factors[1].T @ updated_factors[1], torch.eye(2, dtype=torch.float64))


def test_exact_als_hooi_and_sketches_share_hosvd_initialization():
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64)
    hashes = []
    for method in h05.METHODS:
        core, factors, rows, stats = h05.factorize(tensor, common.SIGNAL_RANKS, method, 11,
                                                  sketch_rows=64, max_sweeps=1)
        hashes.append(stats["initialization_hashes"])
        assert tuple(core.shape) == common.SIGNAL_RANKS
        for mode, factor in enumerate(factors):
            assert factor.shape == (tensor.shape[mode], common.SIGNAL_RANKS[mode])
            torch.testing.assert_close(factor.T @ factor, torch.eye(factor.shape[1], dtype=tensor.dtype), atol=1e-12, rtol=1e-12)
        assert torch.isfinite(common.reconstruct(core, factors)).all()
        if method != "hooi":
            assert len(rows) == 4
            assert all(row["qr_reconstruction_error"] < 1e-10 for row in rows)
    assert all(item == hashes[0] for item in hashes)


def test_sketch_refresh_policy_and_s_less_than_rank_control():
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64)
    _, _, fixed, _ = h05.factorize(tensor, common.SIGNAL_RANKS, "fixed_tensorsketch", 11,
                                   sketch_rows=64, max_sweeps=2, tolerance=-1)
    _, _, fresh, _ = h05.factorize(tensor, common.SIGNAL_RANKS, "refreshed_tensorsketch", 11,
                                   sketch_rows=64, max_sweeps=2, tolerance=-1)
    assert fixed[0]["sketch_hash"] == fixed[4]["sketch_hash"]
    assert fresh[0]["sketch_hash"] != fresh[4]["sketch_hash"]
    with pytest.raises(ValueError, match="s < modal rank"):
        h05.factorize(tensor, common.SIGNAL_RANKS, "refreshed_tensorsketch", 11, sketch_rows=2, max_sweeps=1)


def test_native_synthetic_cnn_shapes_parameters_and_final_split_gate():
    teacher, splits, signal = common.synthetic_cnn(11, include_test=False)
    assert sum(parameter.numel() for parameter in teacher.parameters()) == 1316
    assert "test" not in splits
    assert signal.shape == common.SHAPE
    assert signal.norm().item() == pytest.approx(1, abs=1e-6)
    core, factors = common.exact_hosvd(teacher[2].weight, common.SIGNAL_RANKS)
    student = common.student_from(teacher, core, factors)
    assert sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad) == common.parameter_count(core, factors) + 16
    assert all(not parameter.requires_grad for parameter in student[0].parameters())
    assert all(not parameter.requires_grad for parameter in student[6].parameters())
    assert set(student[2].state_dict()) == {"core", "bias", "factors.0", "factors.1", "factors.2", "factors.3"}
