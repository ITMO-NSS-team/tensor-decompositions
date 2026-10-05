"""CANDECOMP/PARAFAC with the validated NumPy/PyTorch boundary."""
import math
import numbers

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor, CPTensor
from tensorly.decomposition import CP

from tdecomp._base import AbstractDecomposer, TensorDecomposer, _validate_tensor
from tdecomp._random import normalize_random_state, rng_integers
from tdecomp.matrix.random_projections import ProjectorGenerator

__all__ = ['CPDecomposition']


def _options(n_iter_max, tol, init, normalize_factors, linesearch):
    if isinstance(n_iter_max, bool) or not isinstance(n_iter_max, numbers.Integral):
        raise TypeError('n_iter_max must be a positive integer')
    if n_iter_max <= 0:
        raise ValueError('n_iter_max must be a positive integer')
    if tol is not None:
        if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
            raise TypeError('tol must be a nonnegative finite number or None')
        if not math.isfinite(tol) or tol < 0:
            raise ValueError('tol must be a nonnegative finite number or None')
    if not isinstance(init, str) or init not in ('random', 'svd'):
        raise ValueError("init must be 'random' or 'svd'")
    if not isinstance(normalize_factors, bool) or not isinstance(linesearch, bool):
        raise TypeError('normalize_factors and linesearch must be booleans')
    return dict(n_iter_max=int(n_iter_max), tol=tol, init=init,
                normalize_factors=normalize_factors, linesearch=linesearch)


class CPDecomposition(TensorDecomposer):
    """TensorLy 0.9 CP via alternating least squares.

    A positive integer rank is the number of CP components, without a modal
    size cap. A fraction in (0,1] uses floor(rank * min(shape)), at least one;
    None defaults to min(shape). Legacy rank sequences must have one entry
    or one equal entry per mode. Unlike Tucker, CP has one shared rank.

    decompose returns (weights, factors), with weights (rank,) and each
    factor (mode_size,rank). All arrays keep the input backend, dtype and
    device. ALS is nonconvex and need not find an optimal approximation.
    random_init is retained for constructor compatibility; TensorLy controls
    initialization through init. None creates a local stream; integers create
    reproducible streams.

    A positive tol stops when the direct relative reconstruction error is at
    most tol, or its absolute change between ALS steps is below tol. This
    requires one dense reconstruction per step and errors_ records those direct
    residuals, including line-search steps. tol=0 or None disables tolerance
    stopping and retains TensorLy's error history. A small change in error does
    not certify an optimal CP fit.
    """
    def __init__(self, rank=None, random_init=ProjectorGenerator.normal,
                 n_iter_max=100, tol=1e-6, init='random',
                 normalize_factors=False, linesearch=False, random_state=None):
        rng = np.random.default_rng() if random_state is None else random_state
        super().__init__(rank=rank, random_init=random_init, random_state=rng)
        for name, value in _options(n_iter_max, tol, init, normalize_factors, linesearch).items():
            setattr(self, name, value)

    def _get_rank(self, tensor, rank):
        _validate_tensor(tensor)
        shape = tl.shape(tensor)
        if len(shape) < 2:
            raise ValueError('CP requires an input with at least two dimensions')
        selected = self.rank if rank is None else rank
        if selected is None:
            return min(shape)
        if isinstance(selected, (list, tuple, np.ndarray)):
            if np.ndim(selected) != 1 or len(selected) not in (1, len(shape)):
                raise ValueError('CP rank sequence must have one entry or one entry per mode')
            normalized = [self._component_rank(value, min(shape)) for value in selected]
            if any(value != normalized[0] for value in normalized[1:]):
                raise ValueError('CP rank sequence must specify the same component count for every mode')
            return normalized[0]
        return self._component_rank(selected, min(shape))

    @staticmethod
    def _component_rank(rank, dimension):
        if isinstance(rank, bool):
            raise TypeError('Boolean rank is not supported')
        if isinstance(rank, numbers.Integral):
            if rank <= 0:
                raise ValueError('Integer rank must be positive')
            return int(rank)
        if isinstance(rank, numbers.Real):
            if not math.isfinite(rank) or not 0 < rank <= 1:
                raise ValueError('Fractional rank must lie in (0,1]')
            return max(1, int(rank * dimension))
        raise TypeError('CP rank must be a positive integer or a fraction in (0,1]')

    def _decompose(self, tensor, rank, **kwargs):
        names = ('n_iter_max', 'tol', 'init', 'normalize_factors', 'linesearch')
        unknown = set(kwargs) - set(names) - {'random_state'}
        if unknown:
            raise TypeError(f'Unsupported CP options: {sorted(unknown)}')
        options = _options(**{name: kwargs.get(name, getattr(self, name)) for name in names})
        rng = self.random_state if kwargs.get('random_state') is None else normalize_random_state(kwargs['random_state'])
        if not np.any(tl.to_numpy(tensor)):
            # Zero input makes the ALS normal equations singular.
            weights = tl.zeros((rank,), **tl.context(tensor))
            factors = [tl.ones((size, rank), **tl.context(tensor)) for size in tl.shape(tensor)]
            if options['normalize_factors']:
                factors = [factor / math.sqrt(size) for factor, size in zip(factors, tl.shape(tensor))]
            cp_tensor = CPTensor((weights, factors))
            errors, iterations = [], 0
        else:
            callbacks = 0
            direct_errors = []
            tolerance = None if options['tol'] is None else float(options['tol'])
            check_convergence = tolerance is not None and tolerance > 0
            if check_convergence:
                # Scaling avoids overflow/underflow in the squared norm without
                # changing the ALS input or factor dtype/device.
                error_scale = tl.max(tl.abs(tensor))
                scaled_norm = tl.norm(tensor / error_scale, order=2)

            def count_iteration(cp_tensor, error):
                nonlocal callbacks
                callbacks += 1
                if check_convergence and callbacks > 1:
                    # TensorLy's fast norm identity subtracts almost equal
                    # squares near an exact fit. In float32 its error can jump
                    # by O(sqrt(eps)), even when the true residual is O(eps).
                    residual = (tensor - cp_to_tensor(cp_tensor)) / error_scale
                    current_error = float(tl.norm(residual, order=2) / scaled_norm)
                    previous_error = direct_errors[-1] if direct_errors else None
                    direct_errors.append(current_error)
                    return current_error <= tolerance or (
                        previous_error is not None
                        and abs(previous_error - current_error) < tolerance
                    )

            # TensorLy accepts RandomState/int, rather than NumPy Generator.
            seed = int(rng_integers(rng, 0, 2**31 - 1))
            solver_options = dict(options)
            if check_convergence:
                # Only the direct residual may trigger tolerance stopping;
                # TensorLy still calculates its errors for line-search choices.
                solver_options['tol'] = 0
            solver = CP(rank=rank, random_state=seed, callback=count_iteration, **solver_options)
            cp_tensor = solver.fit_transform(tensor)
            errors = direct_errors if check_convergence else [float(error) for error in solver.errors_]
            # TensorLy calls back for initialization and after every ALS step,
            # including line-search steps omitted from solver.errors_.
            iterations = callbacks - 1
        if any(not np.isfinite(tl.to_numpy(value)).all() for value in (cp_tensor.weights, *cp_tensor.factors)):
            raise ArithmeticError('CP factorization produced nonfinite values')
        self.weights_, self.factors_ = cp_tensor.weights, cp_tensor.factors
        self.cp_tensor_ = cp_tensor
        self.errors_ = errors
        self.n_iterations_ = iterations
        return self.weights_, self.factors_

    def compose(self, weights, *factors):
        if len(factors) == 1 and isinstance(factors[0], (list, tuple)):
            factors = factors[0]
        return cp_to_tensor((weights, list(factors)))

    def get_approximation_error(self, tensor, weights=None, *factors, relative=True):
        if weights is None and not factors:
            if not hasattr(self, 'cp_tensor_'):
                raise RuntimeError('Decompose a tensor before requesting the cached CP approximation')
            weights, factors = self.weights_, (self.factors_,)
        return AbstractDecomposer.get_approximation_error(self, tensor, weights, *factors, relative=relative)
