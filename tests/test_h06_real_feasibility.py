import itertools
import math

import pytest
import torch

from experiments.hypotheses import run_h06_real_feasibility as audit


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_real_grids_are_protocol_grids_and_reserve_is_unique():
    first = audit.real_rank_grid("layer2.1.conv2")
    second = audit.real_rank_grid("layer3.0.conv2")
    assert len(first) == len(set(first)) == 225
    assert set(first) == set(itertools.product((16, 32, 64, 96, 128), (16, 32, 64, 96, 128), (1, 2, 3), (1, 2, 3)))
    assert len(second) == len(set(second)) == 145
    assert second[-1] == (256, 256, 3, 3)
    assert (128, 128, 3, 3) not in second


def test_tuple_indexed_unfoldings_have_the_recorded_original_spectra():
    tensor = torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(921), dtype=torch.float64)
    result = audit.modal_spectra(tensor)
    for mode, recorded in enumerate(result["modes"]):
        rest = [k for k in range(4) if k != mode]
        columns = list(itertools.product(*(range(tensor.shape[k]) for k in rest)))
        matrix = torch.empty(tensor.shape[mode], len(columns), dtype=torch.float64)
        for row in range(tensor.shape[mode]):
            for column, coordinates in enumerate(columns):
                ix = [0] * 4
                ix[mode] = row
                for axis, value in zip(rest, coordinates):
                    ix[axis] = value
                matrix[row, column] = tensor[tuple(ix)]
        singular = torch.linalg.svd(matrix, full_matrices=False).S
        torch.testing.assert_close(torch.tensor(recorded["singular_values"], dtype=torch.float64), singular,
                                   atol=1e-13, rtol=1e-13)
        assert recorded["relative_energy_identity_error"] < 1e-13


def test_lower_bound_is_maximum_not_sum_and_does_not_certify_a_candidate():
    tensor = torch.zeros(2, 2, 2, 1, dtype=torch.float64)
    tensor[0, 0, 0, 0] = 1
    tensor[1, 1, 0, 0] = tensor[1, 0, 1, 0] = tensor[0, 1, 1, 0] = .1
    result = audit.modal_spectra(tensor)
    evidence = audit.rank_evidence(result, (1, 1, 1, 1), epsilon=.15)
    expected_tail = 2 * .1**2
    assert evidence["unavoidable_residual_squared_lower_bound"] == pytest.approx(expected_tail)
    assert sum(evidence["modal_tail_squared"]) == pytest.approx(3 * expected_tail)
    assert evidence["relative_error_lower_bound"] < .15
    assert evidence["status"] == "not_excluded_not_certified"
    assert evidence["tucker_candidate_constructed"] is False
    # Its leading-mode HOSVD projection has a larger direct residual. A lower
    # bound passing a threshold cannot certify even that explicit projection.
    approximate = torch.zeros_like(tensor)
    approximate[0, 0, 0, 0] = 1
    direct = float((tensor - approximate).norm() / tensor.norm())
    assert direct > .15


def test_any_rank_one_tucker_matrix_residual_exceeds_unavoidable_tail():
    # Embedding a diagonal matrix in four modes gives analytically known
    # singular values [3,2,1], and a rank-one Tucker matrix has rank <=1.
    tensor = torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64)).reshape(3, 3, 1, 1)
    evidence = audit.rank_evidence(audit.modal_spectra(tensor), (1, 1, 1, 1))
    assert evidence["unavoidable_residual_squared_lower_bound"] == pytest.approx(5)
    assert evidence["relative_error_lower_bound"] == pytest.approx(math.sqrt(5 / 14))
    assert evidence["status"] == "excluded_by_necessary_lower_bound"
    for seed in range(4):
        rng = torch.Generator().manual_seed(seed)
        left, right = torch.randn(3, generator=rng, dtype=torch.float64), torch.randn(3, generator=rng, dtype=torch.float64)
        residual = float((tensor[:, :, 0, 0] - left[:, None] * right[None, :]).square().sum())
        assert residual >= evidence["unavoidable_residual_squared_lower_bound"] - 1e-12
    optimal = torch.zeros_like(tensor)
    optimal[0, 0, 0, 0] = 3
    assert float((tensor - optimal).square().sum()) == pytest.approx(5)


def test_zero_scale_and_full_rank_reserve_semantics():
    tensor = torch.diag(torch.tensor([4., 1.], dtype=torch.float64)).reshape(2, 2, 1, 1)
    reference = audit.rank_evidence(audit.modal_spectra(tensor), (1, 1, 1, 1))
    for scale in (1e-6, 1e6):
        actual = audit.rank_evidence(audit.modal_spectra(scale * tensor), (1, 1, 1, 1))
        assert actual["relative_error_lower_bound"] == pytest.approx(reference["relative_error_lower_bound"])
        assert actual["status"] == reference["status"]
    zero = audit.rank_evidence(audit.modal_spectra(torch.zeros(2, 2, 1, 1)), (1, 1, 1, 1))
    assert zero["relative_error_lower_bound"] is None
    assert zero["status"] == "zero_tensor_absolute_error_zero_relative_undefined"
    full = audit.rank_evidence(audit.modal_spectra(tensor), (2, 2, 1, 1))
    assert full["relative_error_lower_bound"] == 0
    assert full["status"] == "exact_full_rank_reserve"
    assert not full["mathematical_storage_reduction"]


def test_wide_spatial_svd_allocates_only_values_and_never_queries_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU feasibility audit must not call CUDA or allocate SVD bases")
    for name in ("is_available", "current_device", "mem_get_info", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    tensor = torch.randn(16, 12, 3, 3)
    spectra = audit.modal_spectra(tensor, audit.CPUGuard())
    assert spectra["modes"][2]["unfolding_shape"] == [3, 576]
    assert len(spectra["modes"][2]["singular_values"]) == 3
    assert spectra["spectrum_dtype"] == "torch.float64"


def test_baseline_metadata_and_checkpoint_hash_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: None)
    manifest = {"state": "completed", "seeds": list(audit.SEEDS), "epochs": 30,
                "final_test_opened": False, "git_sha": "approved", "data_archive_sha256": "a", "split_sha256": "b"}
    audit.write_json(tmp_path / "manifest.json", manifest)
    for seed in audit.SEEDS:
        directory = tmp_path / f"seed-{seed}"
        directory.mkdir()
        (directory / "model.pt").write_bytes(bytes([seed % 256]))
        audit.write_json(directory / "result.json", {"state": "baseline_ready", "seed": seed,
                         "epochs": 30, "final_test_opened": False,
                         "final_tuning": {"n": 3000, "accuracy": .8},
                         "checkpoint_sha256": audit.file_hash(directory / "model.pt")})
    assert audit.baseline_admission(tmp_path)["dataset_files_read"] is False
    (tmp_path / "seed-202" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.baseline_admission(tmp_path)


@pytest.mark.parametrize("ranks", [(0, 1, 1, 1), (3, 1, 1, 1), (1, 1, 1), (1., 1, 1, 1)])
def test_invalid_rank_contract_is_rejected(ranks):
    with pytest.raises(ValueError, match="integer ranks"):
        audit.rank_evidence(audit.modal_spectra(torch.ones(2, 2, 1, 1)), ranks)
