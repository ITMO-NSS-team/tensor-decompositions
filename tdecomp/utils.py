"""Numerical helpers with explicit domains and backend-preserving results."""
from functools import wraps
from inspect import Parameter, signature
import math
import numbers
import warnings
import numpy as np
import torch
import tensorly as tl
from tdecomp.types import TensorLike
from tdecomp._random import normalize_random_state

__all__ = ['filter_kw_universal', 'filter_kwargs', 'conjugate_gradient', 'svd_solver_tikhonov',
           'pseudo_inverse', 'randperm', 'multinomial', 'topk_ids', 'no_grad', 'svdvals',
           'bool_mask', 'is_complex', 'is_floating_point', 'numel']


def filter_kwargs(func, params):
    """Copy accepted keywords using a callable's public signature."""
    parameters = signature(func).parameters
    if any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(params)
    names = {name for name, p in parameters.items()
             if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY) and name not in {'self', 'cls'}}
    return {name: value for name, value in params.items() if name in names}


def filter_kw_universal(f):
    """Legacy parameter dictionaries are filtered; conventional kwargs checked."""
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return f(self, **filter_kwargs(f, args[0]))
        if 'params' in kwargs and len(kwargs) == 1:
            return f(self, *args, **filter_kwargs(f, kwargs['params']))
        return f(self, *args, **kwargs)
    return wrapped


def _count(k, name='k'):
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(k)


def _tolerance(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def conjugate_gradient(A, b, precond=None, x0=None, max_iter=100, tol=1e-6,
                       verbose=False, device=None, *, rtol=0.0, return_info=False):
    """Torch CG/PCG for SPD/HPD operators, with true residual checks.

    precond applies the inverse preconditioner and can be a matrix or callable.
    tol is absolute, rtol relative to ||b||. Invalid curvature raises ValueError;
    exhausted iterations raise RuntimeError unless return_info=True, which
    returns (x,residuals,diagnostics) with converged=False.
    """
    max_iter = _count(max_iter, 'max_iter')
    tol, rtol = _tolerance(tol, 'tol'), _tolerance(rtol, 'rtol')
    if not isinstance(b, torch.Tensor) or b.ndim != 1 or b.numel() == 0:
        raise ValueError('b must be a nonempty torch vector')
    if b.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128) or not torch.isfinite(b).all():
        raise ValueError('b must have a finite float32/64 or complex64/128 dtype')
    if device is not None and torch.device(device) != b.device:
        raise ValueError('device must match b.device; move inputs explicitly')
    n = b.numel()
    for name, operator in (('A', A), ('precond', precond)):
        if operator is None or callable(operator):
            continue
        if not isinstance(operator, torch.Tensor) or operator.shape != (n, n) or operator.device != b.device or operator.dtype != b.dtype:
            raise ValueError(f'{name} must match b shape, dtype and device')
        if not torch.isfinite(operator).all() or not torch.allclose(operator, operator.mH):
            raise ValueError(f'{name} must be finite and Hermitian')
    if x0 is not None and (x0.shape != b.shape or x0.dtype != b.dtype or x0.device != b.device or not torch.isfinite(x0).all()):
        raise ValueError('x0 must match b and be finite')
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    def apply(operator, vector):
        value = operator(vector) if callable(operator) else operator @ vector
        if not isinstance(value, torch.Tensor) or value.shape != b.shape or value.dtype != b.dtype or value.device != b.device or not torch.isfinite(value).all():
            raise ValueError('Operator must return a finite vector matching b')
        return value

    def positive(value, name):
        scale = abs(value.real.item())
        if not torch.isfinite(value) or abs(value.imag.item() if value.is_complex() else 0) > 100 * torch.finfo(b.real.dtype).eps * scale or value.real.item() <= 0:
            raise ValueError(f'{name} lost positive Hermitian curvature')
        return value.real

    threshold = max(tol, rtol * torch.linalg.vector_norm(b).item())
    r = b - apply(A, x)
    residuals = [torch.linalg.vector_norm(r).item()]
    if verbose:
        print(f'Iter 0: Residual = {residuals[-1]:.3e}')
    converged = residuals[-1] <= threshold
    if not converged and max_iter:
        z = r.clone() if precond is None else apply(precond, r)
        rho = positive(torch.vdot(r, z), 'Preconditioner')
        p = z.clone()
        for iteration in range(1, max_iter + 1):
            ap = apply(A, p)
            curvature = positive(torch.vdot(p, ap), 'Operator')
            x = x + (rho / curvature) * p
            r = b - apply(A, x)
            residuals.append(torch.linalg.vector_norm(r).item())
            if verbose:
                print(f'Iter {iteration}: Residual = {residuals[-1]:.3e}')
            if residuals[-1] <= threshold:
                converged = True
                break
            z = r.clone() if precond is None else apply(precond, r)
            rho_new = positive(torch.vdot(r, z), 'Preconditioner')
            p = z + (rho_new / rho) * p
            rho = rho_new
    info = {'converged': converged, 'iterations': len(residuals)-1,
            'residual_norm': residuals[-1], 'threshold': threshold,
            'reason': 'tolerance' if converged else 'max_iter'}
    if return_info:
        return x, residuals, info
    if not converged:
        raise RuntimeError(f"CG did not converge: residual={residuals[-1]:.6g}, threshold={threshold:.6g}")
    return x, residuals


def _adjoint(matrix):
    return tl.conj(tl.transpose(matrix))


def _matrix_input(matrix):
    if tl.ndim(matrix) != 2 or min(tl.shape(matrix)) <= 0:
        raise ValueError('Expected a nonempty matrix')
    if tl.get_backend() not in ('numpy', 'pytorch'):
        raise NotImplementedError('Matrix solvers support numpy and pytorch backends')
    dtype = str(tl.context(matrix).get('dtype', '')).removeprefix('torch.')
    if dtype not in ('float32', 'float64', 'complex64', 'complex128'):
        raise TypeError('Matrix dtype must be float32, float64, complex64, or complex128')
    values = np.asarray(tl.to_numpy(matrix))
    if not np.isfinite(values).all():
        raise ValueError('Matrix must be finite')
    return values


def pseudo_inverse(A, rcond=None):
    """Moore-Penrose inverse via full thin SVD; threshold=rcond*max(S).

    Default rcond=max(shape)*eps(real dtype). Works for tall, wide, zero,
    rank-deficient and complex matrices without forming normal equations.
    """
    values = _matrix_input(A)
    if rcond is None:
        rcond = max(tl.shape(A)) * np.finfo(values.real.dtype).eps
    rcond = _tolerance(rcond, 'rcond')
    u, s, vh = tl.truncated_svd(A, n_eigenvecs=min(tl.shape(A)))
    keep = s > rcond * tl.max(s)
    safe = tl.where(keep, s, tl.ones(tl.shape(s), **tl.context(s)))
    reciprocal = tl.where(keep, 1 / safe, tl.zeros_like(s))
    inverse = tl.matmul(_adjoint(vh) * reciprocal, _adjoint(u))
    if not np.isfinite(tl.to_numpy(inverse)).all():
        raise ArithmeticError('Pseudoinverse cannot be represented in the matrix dtype')
    return inverse


def svd_solver_tikhonov(A, b, svd_func=None, tol=1e-6, maxiter=20, *, lam=0.0, rtol=0.0, return_info=False):
    """Solve min ||Ax-b||^2 + lam^2||x||^2 in one full-SVD evaluation.

    lam=0 gives the thresholded least-squares/minimum-norm solution. lam is
    explicitly chosen by the caller; no scale-dependent hidden decay runs.
    tol/rtol check the data residual, which may stay nonzero for regularized
    or inconsistent systems. return_info returns (x, diagnostics); otherwise
    an unmet tolerance emits RuntimeWarning. maxiter is retained for source
    compatibility, validated positive, and does not change this closed form.
    A custom svd_func must return a complete thin SVD of A, not a rank-truncated
    approximation. Diagnostics do not claim optimal regularization for noise.
    """
    values = _matrix_input(A)
    tol, rtol, lam = _tolerance(tol, 'tol'), _tolerance(rtol, 'rtol'), _tolerance(lam, 'lam')
    if _count(maxiter, 'maxiter') == 0:
        raise ValueError('maxiter must be positive')
    if tl.context(b).get('dtype') != tl.context(A).get('dtype') or tl.context(b).get('device') != tl.context(A).get('device'):
        raise ValueError('b must match A dtype and device')
    if tl.shape(b) != (tl.shape(A)[0],) or not np.isfinite(tl.to_numpy(b)).all():
        raise ValueError('b must be a finite vector with one value per row')
    u, s, vh = (tl.truncated_svd(A, n_eigenvecs=min(tl.shape(A))) if svd_func is None else svd_func(A))
    rank = min(tl.shape(A))
    if tl.shape(u) != (tl.shape(A)[0], rank) or tl.shape(s) != (rank,) or tl.shape(vh) != (rank, tl.shape(A)[1]):
        raise ValueError('svd_func must return a complete thin SVD')
    if svd_func is not None:
        try:
            contexts = [tl.context(factor) for factor in (u, s, vh)]
        except (TypeError, AttributeError) as exc:
            raise ValueError('svd_func factors must use the same backend as A') from exc
        expected_real = str(values.real.dtype)
        if (contexts[0].get('dtype') != tl.context(A).get('dtype') or
                contexts[2].get('dtype') != tl.context(A).get('dtype') or
                str(contexts[1].get('dtype')).removeprefix('torch.') != expected_real or
                any(c.get('device') != tl.context(A).get('device') for c in contexts)):
            raise ValueError('svd_func factors must match A dtype, real spectrum and device')
        spectra = np.asarray(tl.to_numpy(s))
        if (not all(np.isfinite(tl.to_numpy(factor)).all() for factor in (u,s,vh)) or
                np.any(spectra < 0) or np.any(spectra[1:] > spectra[:-1])):
            raise ValueError('svd_func spectrum must be finite, nonnegative, and descending')
        precision = 100 * max(tl.shape(A)) * np.finfo(values.real.dtype).eps
        if (not np.allclose(tl.to_numpy(tl.matmul(_adjoint(u),u)), np.eye(rank), atol=precision, rtol=precision) or
                not np.allclose(tl.to_numpy(tl.matmul(vh,_adjoint(vh))), np.eye(rank), atol=precision, rtol=precision)):
            raise ValueError('svd_func must return orthonormal singular vectors')
        reconstruction = tl.matmul(u * s, vh)
        scale = max(float(np.max(np.abs(values))), np.finfo(values.real.dtype).tiny)
        if np.linalg.norm(tl.to_numpy(A-reconstruction) / scale) > precision:
            raise ValueError('svd_func must reconstruct A to dtype precision')
    if lam == 0:
        keep = s > max(tl.shape(A)) * np.finfo(values.real.dtype).eps * tl.max(s)
        safe = tl.where(keep, s, tl.ones(tl.shape(s), **tl.context(s)))
        filt = tl.where(keep, 1 / safe, tl.zeros_like(s))
    else:
        # Per-singular-value scaling avoids lambda/global_max underflow at
        # zero modes. Double intermediates also keep tiny lambda representable
        # for float32 input; the final filter retains the original real dtype.
        if tl.get_backend() == 'pytorch':
            work = s.to(torch.float64)
            scale = torch.maximum(work, torch.full_like(work, lam))
            ratio = work / scale
            filt = ((ratio / (ratio**2 + (lam/scale)**2)) / scale).to(s.dtype)
        else:
            work = np.asarray(s, dtype=np.float64)
            scale = np.maximum(work, lam)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                ratio = work / scale
                filt = ((ratio / (ratio**2 + (lam/scale)**2)) / scale).astype(s.dtype)
    if not np.isfinite(tl.to_numpy(filt)).all():
        raise ArithmeticError('Tikhonov filter cannot be represented in the input dtype')
    x = tl.matmul(_adjoint(vh), filt * tl.matmul(_adjoint(u), b))
    if not np.isfinite(tl.to_numpy(x)).all():
        raise ArithmeticError('Tikhonov solution cannot be represented in the input dtype')
    def stable_norm(vector):
        v = np.asarray(tl.to_numpy(vector), dtype=np.complex128 if is_complex(vector) else np.float64)
        scale = float(np.max(np.abs(v)))
        return float(np.linalg.norm(v / scale) * scale) if scale else 0.0
    residual = stable_norm(tl.matmul(A, x)-b)
    if not math.isfinite(residual):
        raise ArithmeticError('Tikhonov residual cannot be represented')
    threshold = max(tol, rtol * stable_norm(b))
    info = {'converged': residual <= threshold, 'residual_norm': residual,
            'threshold': threshold, 'lambda': lam, 'iterations': 1,
            'reason': 'tolerance' if residual <= threshold else 'residual_above_tolerance'}
    if return_info:
        return x, info
    if not info['converged']:
        warnings.warn('Tikhonov data residual exceeds the requested tolerance; use return_info=True for diagnostics', RuntimeWarning, stacklevel=2)
    return x


def index_tensor(values, context=None):
    context = dict(context or {})
    if tl.get_backend() == 'pytorch':
        return torch.as_tensor(values, dtype=torch.long, device=context.get('device', 'cpu'))
    if tl.get_backend() == 'numpy':
        return np.asarray(values, dtype=np.int64)
    raise NotImplementedError('Sampling supports numpy and pytorch backends')


def randperm(k, context=None, *, random_state=None):
    """Backend integer permutation; explicit random_state consumes no globals."""
    k = _count(k)
    return index_tensor(normalize_random_state(random_state).permutation(k), context)


def multinomial(weights, k, context=None, *, random_state=None):
    """A-Res weighted sampling without replacement, preserving zero weights.

    Weights are finite/nonnegative and need not be normalized. k may be zero,
    and cannot exceed the positive support. Log keys are scale invariant.
    """
    k = _count(k)
    values = np.asarray(tl.to_numpy(weights))
    if values.ndim != 1 or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('weights must be a finite nonnegative real vector')
    positive = np.flatnonzero(values > 0)
    if k > len(positive):
        raise ValueError('k cannot exceed the number of positive weights')
    if k == 0:
        return index_tensor([], tl.context(weights) if context is None else context)
    rng = normalize_random_state(random_state)
    u = np.maximum(rng.random(len(positive)), np.finfo(float).tiny)
    scaled = values[positive].astype(np.float64) / np.max(values[positive])
    keys = np.log(u) / scaled
    selected = positive[np.argsort(keys, kind='stable')[-k:]]
    return index_tensor(selected, tl.context(weights) if context is None else context)


def topk_ids(x, k):
    k = _count(k)
    if tl.ndim(x) != 1 or k > tl.shape(x)[0]:
        raise ValueError('topk requires a vector and k <= its size')
    return tl.argsort(x, 0)[-k:] if k else tl.argsort(x, 0)[:0]


def is_complex(x):
    return 'complex' in str(tl.context(x)['dtype'])


def is_floating_point(x):
    return 'float' in str(tl.context(x)['dtype'])


def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = tl.get_backend()
        if backend == 'pytorch':
            with torch.no_grad():
                return func(*args, **kwargs)
        if backend == 'tensorflow':
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ImportError('TensorFlow requires the tdecomp[tensorflow] extra') from exc
            result = func(*args, **kwargs)
            return tf.nest.map_structure(lambda v: tf.stop_gradient(v) if tf.is_tensor(v) else v, result)
        if backend == 'numpy':
            return func(*args, **kwargs)
        raise NotImplementedError(f'Backend {backend} does not support no_grad')
    return wrapper


def bool_mask(shape, context=None):
    context = dict(context or {})
    context['dtype'] = tl.tensor([True]).dtype
    return tl.zeros(shape, **context)


def numel(x):
    return math.prod(tl.shape(x))


def svdvals(x):
    if tl.get_backend() == 'pytorch':
        return torch.linalg.svdvals(x)
    return tl.truncated_svd(x, n_eigenvecs=min(tl.shape(x)))[1]
