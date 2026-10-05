import itertools

import pytest
import torch

from experiments.hypotheses import run_h06_synthetic as h06
from experiments.hypotheses import synthetic_tucker_common as common


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_admission_scales_orders_zero_and_finite_search():
    result = h06.admission()
    assert result["adaptive_direct_certificate_scales_and_orders"] == "passed"
    assert result["bounded_grid"] == "passed"
    assert result["zero_relative_error_undefined"] == "passed"


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 3, 1, 0)])
def test_st_hosvd_telescope_matches_independent_full_residual(order):
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64)
    core, factors, discarded = h06.st_hosvd(tensor, common.SIGNAL_RANKS, order)
    reconstructed = torch.einsum("abcd,oa,ib,hc,wd->oihw", core, *factors)
    direct = float((tensor - reconstructed).square().sum())
    assert sum(discarded) == pytest.approx(direct, abs=1e-14)
    evidence = h06.certificate(tensor, core, factors, discarded=discarded)
    assert evidence["energy_difference_unclamped"] == pytest.approx(direct, abs=1e-14)
    assert evidence["sequential_energy_identity_error"] < 1e-14


def test_adaptive_trace_grows_only_on_declared_grid_and_counts_all_cached_trials():
    tensor, _, _ = common.signal_weight(22, dtype=torch.float64)
    core, factors, info = h06.choose(tensor, "adaptive")
    assert info["certificate"]["admissible"]
    trace = info["search_trace"]
    assert len({tuple(entry["ranks"]) for entry in trace}) == len(trace) == info["searched_tuples"]
    assert len(trace) <= 36
    assert all(all(rank in h06.GRID[mode] for mode, rank in enumerate(entry["ranks"])) for entry in trace)
    visited = [trace[0]["ranks"]] + [entry["ranks"] for entry in trace if entry["selected_next"]]
    # The selected marks are attached to cached rows, which can predate the
    # transition; enforce uniqueness and bound rather than assuming row order.
    assert len(set(visited)) == len(visited)
    assert info["factor_parameters"] == common.parameter_count(core, factors)


def test_adaptive_uses_measured_gain_per_extra_coefficient_for_first_transition():
    tensor, _, _ = common.signal_weight(33, dtype=torch.float64)
    _, _, info = h06.adaptive(tensor)
    entries = info["search_trace"]
    initial = entries[0]
    neighbors = entries[1:5]
    assert len(neighbors) == 4
    scored = [(initial["residual_energy"] - entry["residual_energy"]) /
              (entry["parameters"] - initial["parameters"]) for entry in neighbors]
    best = max(range(4), key=lambda index: scored[index])
    assert neighbors[best]["selected_next"]


@pytest.mark.parametrize("method", h06.METHODS)
def test_flat_spectrum_and_no_admissible_fraction_use_visible_full_rank_reserve(method):
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64, flat=True, sigma=0)
    core, factors, info = h06.choose(tensor, method, fraction=.25)
    assert info["certificate"]["admissible"]
    assert info["full_rank_fallback"]
    assert info["ranks"] == tuple(tensor.shape)
    torch.testing.assert_close(common.reconstruct(core, factors), tensor, atol=0, rtol=0)
    assert info["factor_parameters"] > tensor.numel()
    if method == "fractional":
        assert not info["inadmissible_candidate"]["certificate"]["admissible"]


def test_orthogonality_is_required_and_probe_estimates_do_not_replace_certificate():
    tensor, _, _ = common.signal_weight(11, dtype=torch.float64)
    core, factors = common.exact_hosvd(tensor, common.SIGNAL_RANKS)
    probes = h06.probe_diagnostics(tensor, factors, 11)
    assert len(probes) == 4
    assert all(row["probes"] == 16 and not row["certificate"] for row in probes)
    altered = list(factors)
    altered[0] = 2 * altered[0]
    with pytest.raises(ArithmeticError, match="not orthonormal"):
        h06.certificate(tensor, core, altered)


def test_float32_full_rank_near_cancellation_uses_direct_fp64_without_clamping_energy():
    tensor, _, _ = common.signal_weight(11)
    core, factors = h06.full_rank(tensor)
    evidence = h06.certificate(tensor, core, factors)
    assert evidence["near_cancellation"]
    assert evidence["direct_fp64_residual_energy"] == 0
    assert evidence["energy_difference_unclamped"] == 0
    assert evidence["admissible"]


def test_order_and_parameter_count_are_explicit():
    assert h06.stored_coefficients(common.SHAPE, common.SIGNAL_RANKS) == 148
    with pytest.raises(ValueError, match="each tensor mode"):
        h06.validate_order((0, 1, 1, 3))
