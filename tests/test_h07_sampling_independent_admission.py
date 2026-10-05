"""Independent probability and sampled-span checks, including real-rank sizing."""
import pytest
import tensorly as tl
import torch

from experiments.hypotheses import run_h07_synthetic as h07


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_rank_one_contraction_probability_matches_analytic_input_energy():
    out = torch.tensor([1., -.7, .3], dtype=torch.float64)
    inp = torch.tensor([1., 2., .5, -1.], dtype=torch.float64)
    spatial1 = torch.tensor([1., -.3], dtype=torch.float64)
    spatial2 = torch.tensor([.2, 1., -.4], dtype=torch.float64)
    weight = torch.einsum("o,i,h,w->oihw", out, inp, spatial1, spatial2)
    actual = h07.contraction_probabilities(weight, torch.Generator().manual_seed(53))
    exact = (.8 * inp.square()[:, None, None].expand(4, 2, 3).flatten() / inp.square().sum() / 6
             + .2 / 24)
    torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)


def test_full_output_rank_leverage_matches_exact_row_space_projector():
    weight = torch.randn(8, 7, 2, 2, generator=torch.Generator().manual_seed(85), dtype=torch.float64)
    probability, passes = h07.probabilities(weight, "leverage", torch.Generator().manual_seed(103), rank=8)
    matrix = weight.reshape(8, -1)
    projector = matrix.T @ torch.linalg.solve(matrix @ matrix.T, matrix)
    expected = projector.diagonal() / 8
    expected /= expected.sum()
    torch.testing.assert_close(probability, expected, atol=1e-12, rtol=1e-12)
    assert passes == 2


@pytest.mark.parametrize("method", ("uniform", "column_norm", "leverage", "contraction"))
def test_generalized_rank_first_factor_is_in_true_sampled_span(method):
    weight = torch.randn(8, 7, 2, 2, generator=torch.Generator().manual_seed(85), dtype=torch.float64)
    with tl.backend_context("pytorch"):
        core, factors, info = h07.sampled_tucker(weight, method, 97, b=20, ranks=(6, 4, 2, 2))
        reconstructed = tl.tenalg.multi_mode_dot(core, factors)
    columns = weight.reshape(8, -1)[:, info["sampled_columns"]]
    solution = torch.linalg.lstsq(columns, factors[0], driver="gelsd").solution
    torch.testing.assert_close(columns @ solution, factors[0], rtol=1e-10, atol=1e-10)
    for factor in factors:
        torch.testing.assert_close(factor.T @ factor, torch.eye(factor.shape[1], dtype=weight.dtype), rtol=1e-10, atol=1e-10)
    assert tuple(core.shape) == (6, 4, 2, 2)
    assert info["sampled_span_rank"] >= 6
    assert info["relative_tensor_error"] == pytest.approx(float((weight - reconstructed).norm() / weight.norm()))


def test_rank_deficient_duplicate_draw_cannot_manufacture_output_directions():
    weight = torch.zeros(16, 8, 3, 3, dtype=torch.float64)
    flat = weight.reshape(16, -1)
    flat[0, 0] = 1000
    for i in range(1, 16):
        flat[i, i] = 1
    with tl.backend_context("pytorch"), pytest.raises(ArithmeticError, match="sampled column span rank"):
        h07.sampled_tucker(weight, "column_norm", 11, b=16)
