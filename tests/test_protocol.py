from experiments.two_sided_run import Protocol, run_protocol
import numpy as np
import pytest
from tdecomp.api import SVDMethod, compute_svd


def test_small_protocol_uses_identical_inputs_and_same_rank():
    config = Protocol(shapes=((12, 8), (8, 12)), rank=2, true_rank=5, repeats=2, warmup=0)
    result = run_protocol(config)
    assert len(result["records"]) == result["expected_records"] == 12
    for shape in config.shapes:
        for repeat in range(config.repeats):
            rows = [row for row in result["records"] if row["shape"] == list(shape) and row["repeat"] == repeat]
            assert len({row["input_sha256"] for row in rows}) == 1
            assert all(row["status"] == "success" and row["components"] == 2 for row in rows)
            assert all(row["relative_error"] + 1e-12 >= row["optimal_relative_error"] for row in rows)
    assert result["provenance"]["device"] == "cpu"


def test_protocol_records_failure_without_retry_or_lost_success():
    calls = 0
    def executor(X, options):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected failure")
        return compute_svd(X, options)
    result = run_protocol(Protocol(shapes=((8, 6),), rank=2, true_rank=4, repeats=4,
                                   warmup=0, methods=(SVDMethod.EXACT,)), executor=executor)
    assert calls == 4
    assert [row["status"] for row in result["records"]] == ["success", "success", "failure", "success"]
    assert result["records"][2]["repeat"] == 2


@pytest.mark.parametrize("true_rank, noise", [(0, 0), (2, 0), (4, 0.01)])
def test_protocol_zero_exact_rank_and_noisy_spectrum(true_rank, noise):
    config = Protocol(shapes=((6, 8),), rank=2, true_rank=true_rank, noise=noise, repeats=1, warmup=0)
    result = run_protocol(config)
    assert all(row["status"] == "success" for row in result["records"])
