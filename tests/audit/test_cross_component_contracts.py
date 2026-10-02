"""Consumer invariants spanning the portable API and optimizer.

These checks use independent NumPy reconstruction and application RNG snapshots.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
import tensorly as tl
import torch

from tdecomp.api import SVDContractError, SVDMethod, SVDRequest, compute_svd, load_svd, save_svd


def _numpy_rng_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


@pytest.mark.parametrize("method", list(SVDMethod))
def test_portable_request_isolated_from_caller_backend_and_rng(method):
    x = np.diag([7.0, 2.0, 0.0]).astype(np.float64)
    request = SVDRequest(rank=2, method=method, seed=71, oversampling=1)
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    with tl.backend_context("pytorch"):
        first = compute_svd(x, request)
        assert tl.get_backend() == "pytorch"
        compute_svd(torch.tensor(x), SVDRequest(rank=1, method=method, seed=12))
        second = compute_svd(x, request)
        assert tl.get_backend() == "pytorch"
    assert _numpy_rng_equal(np_before, np.random.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())
    np.testing.assert_allclose(first.U, second.U, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose((first.U * first.S) @ first.Vh, x, rtol=1e-12, atol=1e-12)


def test_numpy_and_torch_calls_can_run_in_independent_threads():
    x = np.diag([8.0, 3.0, 0.0]).astype(np.float64)

    def execute(backend):
        previous = tl.backend.current_backend()
        tl.set_backend(backend, local_threadsafe=True)
        try:
            value = x if backend == "numpy" else torch.tensor(x)
            result = compute_svd(value, SVDRequest(rank=2, method=SVDMethod.RANDOMIZED, seed=3))
            assert tl.get_backend() == backend
            return result.reconstruct()
        finally:
            tl.set_backend(previous, local_threadsafe=True)

    caller_backend = tl.get_backend()
    with ThreadPoolExecutor(max_workers=2) as pool:
        numpy_result, torch_result = list(pool.map(execute, ["numpy", "pytorch"]))
    assert tl.get_backend() == caller_backend
    np.testing.assert_allclose(numpy_result, x, atol=1e-12)
    np.testing.assert_allclose(torch_result.numpy(), x, atol=1e-12)


def test_archive_preserves_named_feature_order_and_independent_reconstruction(tmp_path):
    x = np.array([[1.0, 2.0], [3.0, 6.0], [0.0, 0.0]])
    result = compute_svd(x, SVDRequest(rank=1, feature_names=("temperature", "pressure")))
    path = tmp_path / "decomposition.npz"
    save_svd(result, path)
    loaded = load_svd(path, feature_names=("temperature", "pressure"))
    np.testing.assert_allclose((loaded.U * loaded.S) @ loaded.Vh, x, atol=1e-12)
    with pytest.raises(ValueError):
        load_svd(path, feature_names=("pressure", "temperature"))


def test_legacy_modal_result_reconstructs_anisotropic_tensor():
    from tdecomp.output_formats import ModalDecomposition

    core = np.arange(8.0).reshape(2, 2, 2)
    factors = [np.eye(3, 2), np.eye(4, 2), np.eye(5, 2)]
    with tl.backend_context("numpy"):
        actual = ModalDecomposition.compose(core, factors)
    expected = np.einsum("abc,ia,jb,kc->ijk", core, *factors)
    np.testing.assert_array_equal(actual, expected)


def test_archive_rejects_forged_huge_shape_before_materializing_arrays(tmp_path, monkeypatch):
    source = tmp_path / "valid.npz"
    corrupt = tmp_path / "huge-header.npz"
    save_svd(compute_svd(np.eye(2), SVDRequest(rank=1)), source)
    header = BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (2**40, 2**40)}
    )
    with ZipFile(source) as original, ZipFile(corrupt, "w") as modified:
        for name in original.namelist():
            modified.writestr(name, header.getvalue() if name == "U.npy" else original.read(name))

    def forbidden_load(*args, **kwargs):
        pytest.fail("Array materialization must not run before admission of NPY headers")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(SVDContractError) as error:
        load_svd(corrupt)
    assert error.value.code in {"invalid_artifact", "resource_limit"}
