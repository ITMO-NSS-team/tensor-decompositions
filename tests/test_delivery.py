"""Regressions through native package imports and the portable public boundary."""
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from tdecomp.api import (CAPABILITIES, ResourcePolicy, SVDContractError,
                        SVDMethod, SVDRequest, compute_svd, load_svd, save_svd)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_import_preserves_application_state(backend):
    script = '''
import importlib, logging, os, sys
import numpy as np
import torch
import tensorly as tl
tl.set_backend(sys.argv[1])
os.environ['WANDB_MODE'] = 'online'
np.random.seed(32); torch.manual_seed(42)
np_state = np.random.get_state(); torch_state = torch.random.get_rng_state().clone()
env = dict(os.environ); handlers = list(logging.getLogger().handlers)
import tdecomp
importlib.reload(tdecomp)
import tdecomp.matrix, tdecomp.types, tdecomp.utils
assert tl.get_backend() == sys.argv[1]
assert env == dict(os.environ)
assert handlers == list(logging.getLogger().handlers)
assert np.array_equal(np_state[1], np.random.get_state()[1])
assert torch.equal(torch_state, torch.random.get_rng_state())
assert 'tensorflow' not in sys.modules
assert not any(k.startswith('tdecomp.grad_proj') for k in sys.modules)
'''
    subprocess.run([sys.executable, "-c", script, backend], check=True)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_mask_context_is_not_mutated(backend):
    import tensorly as tl
    from tdecomp.utils import bool_mask
    with tl.backend_context(backend):
        context = {"dtype": tl.float64}
        original = dict(context)
        mask = bool_mask((3, 4), context)
        assert context == original
        assert str(mask.dtype) in ("bool", "torch.bool")
        assert not mask.any()


def test_native_tensorflow_no_grad():
    tf = pytest.importorskip("tensorflow")
    import tensorly as tl
    from tdecomp.utils import no_grad
    with tl.backend_context("tensorflow"):
        @no_grad
        def operation(x):
            return {"values": [x * x], "name": "tensor"}
        x = tf.constant([2.0, 3.0])
        with tf.GradientTape() as tape:
            tape.watch(x)
            result = operation(x)
            loss = tf.reduce_sum(result["values"][0])
        assert tape.gradient(loss, x) is None
        assert result["name"] == "tensor"


@pytest.mark.parametrize("shape", [(7, 3), (3, 7), (4, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_exact_public_svd_matches_truncated_reference(shape, dtype, backend):
    X = np.random.default_rng(31).normal(size=shape).astype(dtype)
    original = X.copy()
    matrix = X if backend == "numpy" else torch.from_numpy(X.copy())
    result = compute_svd(matrix, SVDRequest(rank=2))
    approximate = result.reconstruct()
    approximate = approximate.numpy() if backend == "pytorch" else approximate
    tail = np.linalg.svd(X, compute_uv=False)[2:]
    assert np.isclose(np.linalg.norm(X - approximate), np.linalg.norm(tail), rtol=2e-5, atol=1e-5)
    assert result.U.shape == (shape[0], 2)
    assert result.Vh.shape == (2, shape[1])
    assert result.components == result.requested_rank == 2
    assert result.diagnostics.backend == backend
    assert np.array_equal(X, original)


@pytest.mark.parametrize("method", list(SVDMethod))
def test_seed_repeatability_and_zero_matrix(method):
    X = np.random.default_rng(11).normal(size=(9, 5))
    request = SVDRequest(3, method=method, seed=19)
    first = compute_svd(X, request)
    second = compute_svd(X, request)
    assert np.allclose(first.reconstruct(), second.reconstruct())
    zero = compute_svd(np.zeros_like(X), request)
    assert zero.numerical_rank == 0
    assert zero.diagnostics.relative_reconstruction_error == 0
    assert np.array_equal(zero.reconstruct(), X * 0)


@pytest.mark.parametrize("X, options, code", [
    (np.eye(3), SVDRequest(True), "invalid_request"),
    (np.eye(3), SVDRequest(4), "invalid_rank"),
    (np.eye(3), SVDRequest(0), "invalid_request"),
    (np.eye(3), SVDRequest(2, method="exact"), "invalid_request"),
    (np.eye(3), SVDRequest(2, seed=-1), "invalid_request"),
    (np.eye(3, dtype=np.int64), SVDRequest(2), "unsupported_dtype"),
    (np.eye(3, dtype=np.complex128), SVDRequest(2), "unsupported_dtype"),
    (np.zeros((0, 3)), SVDRequest(2), "invalid_shape"),
    (np.ones((2, 3, 4)), SVDRequest(2), "invalid_shape"),
    (np.full((3, 3), np.nan), SVDRequest(2), "nonfinite_input"),
    (torch.eye(3, requires_grad=True), SVDRequest(2), "unsupported_layout"),
    (np.eye(3), SVDRequest(2, resources=ResourcePolicy(max_input_bytes=1)), "resource_limit"),
])
def test_public_contract_rejects_invalid_requests(X, options, code):
    with pytest.raises(SVDContractError) as error:
        compute_svd(X, options)
    assert error.value.code == code


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_artifact_roundtrip_and_feature_projection(tmp_path, backend):
    X = np.random.default_rng(4).normal(size=(8, 3))
    names = ("a", "b", "c")
    fitted = compute_svd(X, SVDRequest(2, feature_names=names))
    target = tmp_path / "result.npz"
    save_svd(fitted, target)
    result = load_svd(target, backend=backend, feature_names=names)
    matrix = X if backend == "numpy" else torch.from_numpy(X)
    scores = result.transform(matrix, feature_names=names)
    assert np.allclose(scores if backend == "numpy" else scores.numpy(), fitted.U * fitted.S)
    assert np.array_equal(result.S if backend == "numpy" else result.S.numpy(), fitted.S)
    with pytest.raises(SVDContractError, match="order"):
        load_svd(target, feature_names=("b", "a", "c"))
    with pytest.raises(SVDContractError) as error:
        result.transform(matrix, feature_names=names, resources=ResourcePolicy(max_output_bytes=1))
    assert error.value.code == "resource_limit"


def test_artifact_rejects_version_and_array_tampering(tmp_path):
    target = tmp_path / "result.npz"
    save_svd(compute_svd(np.eye(3), SVDRequest(2)), target)
    with np.load(target, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["artifact_version"] = 20
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez(target, **arrays)
    with pytest.raises(SVDContractError) as error:
        load_svd(target)
    assert error.value.code == "unsupported_artifact_version"
    metadata["artifact_version"] = 1
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["U"] = arrays["U"] + 1
    np.savez(target, **arrays)
    with pytest.raises(SVDContractError, match="checksum"):
        load_svd(target)


def test_bad_npy_header_rejected_before_allocation(tmp_path):
    target = tmp_path / "result.npz"
    save_svd(compute_svd(np.eye(3), SVDRequest(2)), target)
    with ZipFile(target) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    payload = BytesIO()
    np.lib.format.write_array_header_1_0(payload, {"descr": "<f8", "fortran_order": False, "shape": (2**40, 2)})
    entries["U.npy"] = payload.getvalue()
    with ZipFile(target, "w") as destination:
        for name, data in entries.items():
            destination.writestr(name, data)
    with pytest.raises(SVDContractError, match="stored bytes"):
        load_svd(target)


def test_failed_save_preserves_existing_file(tmp_path):
    result = compute_svd(np.eye(3), SVDRequest(2))
    target = tmp_path / "result.npz"
    save_svd(result, target)
    before = target.read_bytes()
    with pytest.raises(SVDContractError):
        save_svd(replace(result, Vh=np.zeros((1, 3))), target)
    assert target.read_bytes() == before


def test_load_working_budget_checked_before_array_materialization(tmp_path, monkeypatch):
    target = tmp_path / "result.npz"
    save_svd(compute_svd(np.eye(3), SVDRequest(2)), target)
    def unexpected_load(*args, **kwargs):
        raise AssertionError("np.load must not run before working-budget admission")
    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(SVDContractError) as error:
        load_svd(target, resources=ResourcePolicy(max_estimated_working_bytes=1))
    assert error.value.code == "resource_limit"


def test_capability_registry_contains_only_supported_methods():
    assert set(CAPABILITIES) == set(SVDMethod)
    assert all(cap.devices == ("cpu",) and cap.result == "U_S_Vh" for cap in CAPABILITIES.values())
    with pytest.raises(TypeError):
        CAPABILITIES["other"] = None
