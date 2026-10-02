"""Portable dense real CPU SVD boundary, independent of consumer frameworks.

The in-process profile enforces byte admission budgets, not an OS memory limit or
deadline. Applications requiring hard deadlines must supervise a child process.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from types import MappingProxyType
from typing import Literal
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
Backend = Literal["numpy", "pytorch"]
ARTIFACT_VERSION = 1

__all__ = ["SVDMethod", "SVDRequest", "ResourcePolicy", "SVDResult",
           "ExecutionInfo", "MethodCapabilities", "SVDContractError",
           "CAPABILITIES", "compute_svd", "save_svd", "load_svd"]


class SVDMethod(str, Enum):
    EXACT = "exact"
    RANDOMIZED = "randomized"
    TWO_SIDED = "two_sided"


class SVDContractError(ValueError):
    """Expected contract failure with a stable machine-readable code."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ResourcePolicy:
    max_input_bytes: int = 64 * 1024**2
    max_output_bytes: int = 64 * 1024**2
    max_estimated_working_bytes: int = 512 * 1024**2


@dataclass(frozen=True)
class SVDRequest:
    rank: int
    method: SVDMethod = SVDMethod.EXACT
    seed: int = 0
    power: int = 2
    oversampling: int = 5
    feature_names: tuple[str, ...] | None = None
    resources: ResourcePolicy = field(default_factory=ResourcePolicy)


@dataclass(frozen=True)
class MethodCapabilities:
    backends: tuple[str, ...] = ("numpy", "pytorch")
    dtypes: tuple[str, ...] = ("float32", "float64")
    devices: tuple[str, ...] = ("cpu",)
    layout: str = "dense_real_2d"
    result: str = "U_S_Vh"
    local_rng: bool = True


CAPABILITIES = MappingProxyType({method: MethodCapabilities() for method in SVDMethod})


@dataclass(frozen=True)
class ExecutionInfo:
    method: str
    backend: str
    dtype: str
    device: str
    seed: int
    power: int
    oversampling: int
    input_bytes: int
    output_bytes: int
    estimated_working_bytes: int
    factorization_seconds: float
    relative_reconstruction_error: float
    numpy_version: str
    torch_version: str
    tensorly_version: str
    torch_threads: int
    memory_measurement: str = "admission_estimate_not_peak"
    numerical_rank_scope: str = "returned_spectrum"


@dataclass(frozen=True)
class SVDResult:
    U: Array
    S: Array
    Vh: Array
    input_shape: tuple[int, int]
    requested_rank: int
    components: int
    numerical_rank: int
    rank_threshold: float
    feature_names: tuple[str, ...] | None
    diagnostics: ExecutionInfo

    def reconstruct(self) -> Array:
        return (self.U * self.S) @ self.Vh

    def transform(self, X: Array, *, feature_names: tuple[str, ...] | None = None,
                  resources: ResourcePolicy = ResourcePolicy()) -> Array:
        """Project new rows to X @ Vh.T; training scores equal U * S."""
        _validate_result(self)
        _check_resources(resources)
        backend, dtype, shape, input_bytes = _describe_input(X)
        if backend != self.diagnostics.backend or dtype != self.diagnostics.dtype:
            raise SVDContractError("backend_mismatch", "Projection requires the fitted backend and dtype")
        if shape[1] != self.input_shape[1] or feature_names != self.feature_names:
            raise SVDContractError("feature_order_mismatch", "Feature count and ordered names must match fit")
        output_bytes = shape[0] * self.components * np.dtype(dtype).itemsize
        if (input_bytes > resources.max_input_bytes or output_bytes > resources.max_output_bytes
                or 2 * (input_bytes + output_bytes) > resources.max_estimated_working_bytes):
            raise SVDContractError("resource_limit", "Projection exceeds byte admission budget")
        if not _finite(X):
            raise SVDContractError("nonfinite_input", "Projection input must contain finite values")
        return X @ self.Vh.T


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise SVDContractError("invalid_request", f"{name} must be an integer >= {minimum}")
    return int(value)


def _check_resources(policy):
    if not isinstance(policy, ResourcePolicy):
        raise SVDContractError("invalid_request", "resources must be ResourcePolicy")
    for name, value in asdict(policy).items():
        _integer(value, name, minimum=1)


def _names(names, count):
    if names is None:
        return
    if (not isinstance(names, tuple) or len(names) != count
            or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != count):
        raise SVDContractError("feature_order_mismatch", "feature_names must be unique ordered nonempty strings")
    if sum(6 * len(name) + 4 for name in names) > 8192:
        raise SVDContractError("resource_limit", "Ordered feature names exceed artifact metadata budget")


def _describe_input(X):
    if isinstance(X, np.ndarray):
        backend = "numpy"
        if X.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise SVDContractError("unsupported_dtype", "Only float32/float64 are supported")
        dtype, size = X.dtype.name, X.nbytes
    elif isinstance(X, torch.Tensor):
        backend = "pytorch"
        if X.device.type != "cpu":
            raise SVDContractError("unsupported_device", "Only CPU tensors are supported")
        if X.layout != torch.strided or X.requires_grad:
            raise SVDContractError("unsupported_layout", "Dense tensors without autograd are required")
        if X.dtype not in (torch.float32, torch.float64):
            raise SVDContractError("unsupported_dtype", "Only float32/float64 are supported")
        dtype, size = str(X.dtype).split(".")[-1], X.numel() * X.element_size()
    else:
        raise SVDContractError("unsupported_input", "Expected numpy.ndarray or torch.Tensor")
    if X.ndim != 2 or any(d <= 0 for d in X.shape):
        raise SVDContractError("invalid_shape", "Expected nonempty 2D matrix")
    return backend, dtype, tuple(X.shape), size


def _numpy(X):
    return X.detach().numpy() if isinstance(X, torch.Tensor) else X


def _finite(X):
    return bool(np.isfinite(X).all()) if isinstance(X, np.ndarray) else bool(torch.isfinite(X).all())


@contextmanager
def _local_backend(backend):
    # TensorLy's backend_context restores the global default in its finally;
    # restore explicitly in thread-local scope instead.
    import tensorly as tl
    previous = tl.backend.current_backend()
    tl.set_backend(backend, local_threadsafe=True)
    try:
        yield
    finally:
        tl.set_backend(previous, local_threadsafe=True)


def _validate_result(result):
    try:
        _validate_result_fields(result)
    except SVDContractError:
        raise
    except (ValueError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise SVDContractError("invalid_result", "Malformed SVD result") from exc


def _validate_result_fields(result):
    if not isinstance(result, SVDResult) or not isinstance(result.diagnostics, ExecutionInfo):
        raise SVDContractError("invalid_result", "Expected SVDResult with ExecutionInfo")
    if not isinstance(result.input_shape, tuple) or len(result.input_shape) != 2:
        raise SVDContractError("invalid_result", "Invalid input shape")
    for name in ("requested_rank", "components", "numerical_rank"):
        _integer(getattr(result, name), name)
    for dimension in result.input_shape:
        _integer(dimension, "input_shape", minimum=1)
    for array in (result.U, result.S, result.Vh):
        if not isinstance(array, (np.ndarray, torch.Tensor)):
            raise SVDContractError("invalid_result", "Factors must be numeric arrays")
        if isinstance(array, torch.Tensor) and (array.device.type != "cpu" or array.layout != torch.strided or array.requires_grad):
            raise SVDContractError("invalid_result", "Factors must be dense CPU tensors without autograd")
    m, n = result.input_shape
    k = result.components
    if (k != result.requested_rank or not 1 <= k <= min(m, n)
            or tuple(result.U.shape) != (m, k) or tuple(result.S.shape) != (k,)
            or tuple(result.Vh.shape) != (k, n)):
        raise SVDContractError("invalid_result", "Invalid factor shapes or component count")
    arrays = tuple(_numpy(a) for a in (result.U, result.S, result.Vh))
    if any(a.dtype.name != result.diagnostics.dtype or not np.isfinite(a).all() for a in arrays):
        raise SVDContractError("invalid_result", "Invalid factor dtype or nonfinite factor")
    u, s, vh = arrays
    eps = np.finfo(s.dtype).eps
    tolerance = 100 * eps * max(m, n)
    if (np.any(s < 0) or np.any(s[1:] > s[:-1])
            or not np.allclose(u.T @ u, np.eye(k), atol=tolerance, rtol=tolerance)
            or not np.allclose(vh @ vh.T, np.eye(k), atol=tolerance, rtol=tolerance)):
        raise SVDContractError("invalid_result", "SVD factors violate orthogonality or spectrum order")
    if (not np.isfinite(result.rank_threshold) or result.rank_threshold < 0
            or result.numerical_rank != int(np.count_nonzero(s > result.rank_threshold))):
        raise SVDContractError("invalid_result", "Invalid numerical rank or threshold")
    _names(result.feature_names, n)
    info = result.diagnostics
    if (info.backend not in ("numpy", "pytorch") or info.dtype not in ("float32", "float64")
            or info.device != "cpu" or info.method not in {method.value for method in SVDMethod}
            or info.numerical_rank_scope != "returned_spectrum"
            or info.memory_measurement != "admission_estimate_not_peak"):
        raise SVDContractError("invalid_result", "Unsupported execution metadata")
    for name in ("seed", "power", "oversampling", "input_bytes", "output_bytes",
                 "estimated_working_bytes", "torch_threads"):
        _integer(getattr(info, name), name)
    itemsize = np.dtype(info.dtype).itemsize
    if (info.input_bytes != m * n * itemsize
            or info.output_bytes != (m * k + k + k * n) * itemsize
            or info.estimated_working_bytes < info.input_bytes + info.output_bytes
            or info.torch_threads < 1 or info.power > 20 or info.oversampling > 1024):
        raise SVDContractError("invalid_result", "Inconsistent execution metadata")
    for value in (info.factorization_seconds, info.relative_reconstruction_error):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise SVDContractError("invalid_result", "Invalid execution measurement")
    for name in ("numpy_version", "torch_version", "tensorly_version"):
        if not isinstance(getattr(info, name), str) or not getattr(info, name):
            raise SVDContractError("invalid_result", "Invalid dependency version")
    if any(isinstance(a, torch.Tensor) != (info.backend == "pytorch") for a in (result.U, result.S, result.Vh)):
        raise SVDContractError("invalid_result", "Factor backend does not match metadata")


def compute_svd(X: Array, request: SVDRequest) -> SVDResult:
    """Compute requested components without changing input or consumer settings."""
    if not isinstance(request, SVDRequest) or not isinstance(request.method, SVDMethod):
        raise SVDContractError("invalid_request", "Expected SVDRequest and SVDMethod enum")
    _check_resources(request.resources)
    rank = _integer(request.rank, "rank", minimum=1)
    seed = _integer(request.seed, "seed")
    power = _integer(request.power, "power")
    oversampling = _integer(request.oversampling, "oversampling")
    if power > 20 or oversampling > 1024:
        raise SVDContractError("invalid_request", "power <= 20 and oversampling <= 1024 are required")
    backend, dtype, shape, input_bytes = _describe_input(X)
    m, n = shape
    if rank > min(shape):
        raise SVDContractError("invalid_rank", "rank must not exceed the smaller input dimension")
    _names(request.feature_names, n)
    itemsize = np.dtype(dtype).itemsize
    output_bytes = (m * rank + rank + rank * n) * itemsize
    p = min(shape) if request.method == SVDMethod.EXACT else min(min(shape), rank + oversampling)
    estimate = (4 * m * n + 8 * (m + n) * p + 8 * p * p) * max(itemsize, 8)
    if (input_bytes > request.resources.max_input_bytes
            or output_bytes > request.resources.max_output_bytes
            or estimate > request.resources.max_estimated_working_bytes):
        raise SVDContractError("resource_limit", "Request exceeds byte admission budget")
    if not _finite(X):
        raise SVDContractError("nonfinite_input", "Input must contain finite values")
    import tensorly as tl
    start = perf_counter()
    try:
        if request.method == SVDMethod.EXACT:
            if backend == "numpy":
                u, s, vh = np.linalg.svd(X, full_matrices=False)
            else:
                u, s, vh = torch.linalg.svd(X, full_matrices=False)
            if backend == "numpy":
                u, s, vh = u[:, :rank].copy(), s[:rank].copy(), vh[:rank].copy()
            else:
                u, s, vh = u[:, :rank].clone(), s[:rank].clone(), vh[:rank].clone()
        else:
            from .matrix.decomposer import RandomizedSVD, TwoSidedRandomSVD
            with _local_backend(backend):
                cls = RandomizedSVD if request.method == SVDMethod.RANDOMIZED else TwoSidedRandomSVD
                method = cls(rank=rank, random_state=seed, oversampling=oversampling, power=power)
                u, s, vh = method.decompose(X, rank=rank)
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise SVDContractError("numerical_failure", "SVD factorization failed") from exc
    elapsed = perf_counter() - start
    s_np = _numpy(s)
    threshold = float(np.finfo(dtype).eps * max(shape) * (s_np[0] if len(s_np) else 0))
    # Diagnostic reductions use float64; no reconstruction is timed as factorization.
    input64 = _numpy(X).astype(np.float64, copy=False)
    approximation = (_numpy(u).astype(np.float64) * s_np) @ _numpy(vh).astype(np.float64)
    scale = float(np.max(np.abs(input64)))
    error = 0.0 if scale == 0 else float(np.linalg.norm((input64 - approximation) / scale) / np.linalg.norm(input64 / scale))
    if not np.isfinite(error):
        raise SVDContractError("numerical_failure", "Reconstruction diagnostic is nonfinite")
    info = ExecutionInfo(request.method.value, backend, dtype, "cpu", seed, power,
                         oversampling, input_bytes, output_bytes, estimate, elapsed, error,
                         np.__version__, torch.__version__, tl.__version__, torch.get_num_threads())
    result = SVDResult(u, s, vh, shape, rank, rank, int(np.count_nonzero(s_np > threshold)),
                       threshold, request.feature_names, info)
    _validate_result(result)
    return result


def _digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_svd(result: SVDResult, path: str | Path) -> None:
    """Save numeric NPZ arrays and versioned JSON, with no pickle objects."""
    _validate_result(result)
    arrays = {name: _numpy(getattr(result, name)) for name in ("U", "S", "Vh")}
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "layout": "U_S_Vh",
        "input_shape": result.input_shape, "requested_rank": result.requested_rank,
        "components": result.components, "numerical_rank": result.numerical_rank,
        "rank_threshold": result.rank_threshold, "feature_names": result.feature_names,
        "diagnostics": asdict(result.diagnostics),
        "checksums": {name: _digest(a) for name, a in arrays.items()},
    }
    # One file prevents mismatched metadata/array pairs after interrupted writes.
    target = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays, metadata=np.array(json.dumps(metadata, allow_nan=False, sort_keys=True)))
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_svd(path: str | Path, *, backend: Backend = "numpy",
             feature_names: tuple[str, ...] | None = None,
             resources: ResourcePolicy = ResourcePolicy()) -> SVDResult:
    """Load a bounded v1 artifact; output backend conversion must be explicit."""
    _check_resources(resources)
    if backend not in ("numpy", "pytorch"):
        raise SVDContractError("unsupported_backend", "Expected numpy or pytorch")
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            headers = {}
            if len(entries) != 4 or {e.filename for e in entries} != {"U.npy", "S.npy", "Vh.npy", "metadata.npy"}:
                raise SVDContractError("invalid_artifact", "Unexpected or duplicated archive entries")
            if any(e.filename == "metadata.npy" and e.file_size > 65536 for e in entries):
                raise SVDContractError("resource_limit", "Artifact metadata exceeds 64 KiB")
            array_bytes = sum(e.file_size for e in entries if e.filename != "metadata.npy")
            if array_bytes > resources.max_output_bytes + 3 * 1024:
                raise SVDContractError("resource_limit", "Artifact exceeds output byte budget")
            for entry in entries:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape_header, _, dtype_header = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape_header, _, dtype_header = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise SVDContractError("invalid_artifact", "Unsupported NPY header version")
                    if dtype_header.hasobject:
                        raise SVDContractError("invalid_artifact", "Object arrays are prohibited")
                    count = 1
                    for dimension in shape_header:
                        count *= _integer(dimension, "array dimension")
                    required = count * dtype_header.itemsize
                    if required != entry.file_size - stream.tell():
                        raise SVDContractError("invalid_artifact", "NPY dimensions do not match stored bytes")
                    if entry.filename == "metadata.npy":
                        if shape_header != () or dtype_header.kind != "U":
                            raise SVDContractError("invalid_artifact", "Invalid metadata array")
                    elif (dtype_header not in (np.dtype("float32"), np.dtype("float64"))
                            or len(shape_header) not in (1, 2) or required > resources.max_output_bytes):
                        raise SVDContractError("invalid_artifact", "Invalid factor header")
                    headers[entry.filename] = (shape_header, dtype_header, required)
            u_shape, u_dtype, _ = headers["U.npy"]
            s_shape, s_dtype, _ = headers["S.npy"]
            v_shape, v_dtype, _ = headers["Vh.npy"]
            if (len(u_shape) != 2 or len(s_shape) != 1 or len(v_shape) != 2
                    or s_shape[0] < 1 or u_shape[0] < 1 or v_shape[1] < 1
                    or u_shape[1] != s_shape[0] or v_shape[0] != s_shape[0]
                    or s_shape[0] > min(u_shape[0], v_shape[1])
                    or u_dtype != s_dtype or v_dtype != s_dtype):
                raise SVDContractError("invalid_artifact", "Inconsistent factor headers")
            materialized_bytes = sum(headers[name][2] for name in ("U.npy", "S.npy", "Vh.npy"))
            # Include explicit backend copies and orthogonality Gram/identity
            # temporaries. This admission uses inspected headers, not metadata.
            load_estimate = 2 * materialized_bytes + 64 * s_shape[0]**2 + headers["metadata.npy"][2]
            if (materialized_bytes > resources.max_output_bytes
                    or u_shape[0] * v_shape[1] * s_dtype.itemsize > resources.max_input_bytes
                    or load_estimate > resources.max_estimated_working_bytes):
                raise SVDContractError("resource_limit", "Artifact materialization exceeds byte admission budget")
        with np.load(path, allow_pickle=False) as archive:
            raw = archive["metadata"]
            if raw.shape != () or raw.dtype.kind != "U":
                raise SVDContractError("invalid_artifact", "Metadata must be scalar Unicode JSON")
            meta = json.loads(str(raw))
            keys = {"artifact_version", "layout", "input_shape", "requested_rank", "components",
                    "numerical_rank", "rank_threshold", "feature_names", "diagnostics", "checksums"}
            if not isinstance(meta, dict) or set(meta) != keys:
                raise SVDContractError("invalid_artifact", "Unknown or missing metadata fields")
            if type(meta["artifact_version"]) is not int or meta["artifact_version"] != ARTIFACT_VERSION:
                raise SVDContractError("unsupported_artifact_version", "Unsupported artifact version")
            if meta["layout"] != "U_S_Vh":
                raise SVDContractError("invalid_artifact", "Unsupported factor layout")
            shape = meta["input_shape"]
            if not isinstance(shape, list) or len(shape) != 2:
                raise SVDContractError("invalid_artifact", "Invalid input shape")
            shape = tuple(_integer(d, "input_shape", minimum=1) for d in shape)
            if meta["feature_names"] is not None and not isinstance(meta["feature_names"], list):
                raise SVDContractError("invalid_artifact", "Feature names must be an ordered JSON array")
            names = None if meta["feature_names"] is None else tuple(meta["feature_names"])
            _names(names, shape[1])
            if names != feature_names:
                raise SVDContractError("feature_order_mismatch", "Expected feature order does not match artifact")
            info = ExecutionInfo(**meta["diagnostics"])
            if (info.backend not in ("numpy", "pytorch") or info.dtype not in ("float32", "float64")
                    or info.device != "cpu" or info.method not in {method.value for method in SVDMethod}
                    or info.numerical_rank_scope != "returned_spectrum"
                    or info.memory_measurement != "admission_estimate_not_peak"):
                raise SVDContractError("invalid_artifact", "Unsupported execution metadata")
            if (info.input_bytes > resources.max_input_bytes
                    or info.estimated_working_bytes > resources.max_estimated_working_bytes):
                raise SVDContractError("resource_limit", "Artifact exceeds fitted byte admission budget")
            rank = _integer(meta["requested_rank"], "requested_rank", minimum=1)
            components = _integer(meta["components"], "components", minimum=1)
            numeric_rank = _integer(meta["numerical_rank"], "numerical_rank")
            threshold = meta["rank_threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise SVDContractError("invalid_artifact", "Invalid rank threshold")
            arrays = {name: archive[name] for name in ("U", "S", "Vh")}
            if (not isinstance(meta["checksums"], dict) or set(meta["checksums"]) != set(arrays)
                    or any(_digest(a) != meta["checksums"][name] for name, a in arrays.items())):
                raise SVDContractError("invalid_artifact", "Array checksum mismatch")
            if sum(a.nbytes for a in arrays.values()) > resources.max_output_bytes:
                raise SVDContractError("resource_limit", "Artifact exceeds output byte budget")
            if backend == "pytorch":
                arrays = {name: torch.from_numpy(a.copy()) for name, a in arrays.items()}
            info = ExecutionInfo(**(asdict(info) | {"backend": backend}))
            result = SVDResult(arrays["U"], arrays["S"], arrays["Vh"], shape, rank, components,
                               numeric_rank, float(threshold), names, info)
            _validate_result(result)
            return result
    except SVDContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, BadZipFile) as exc:
        raise SVDContractError("invalid_artifact", "Malformed numeric artifact") from exc
