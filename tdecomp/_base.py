"""Validated matrix and Tucker decomposition boundaries.

Integer ranks are positive and capped by mode size. Fractional ranks lie in
(0,1] and use floor with at least one component. They are recomputed on every
call; configured ranks are never replaced by shape-dependent values.
"""
import math
import numbers
from functools import wraps
from abc import ABC, abstractmethod
import numpy as np
import tensorly as tl
from tdecomp.types import Number, TensorLike
from tdecomp._random import normalize_random_state
from tdecomp.matrix.random_projections import ProjectorGenerator

__all__ = ['Number', 'Decomposer', 'TensorDecomposer', 'BaseSketch']
DIM_SUM_LIM = 1024
DIM_LIM = 1024


def _adjoint(x):
    return tl.conj(tl.transpose(x))


def _validate_tensor(x, ndim=None):
    shape = tl.shape(x)
    if ndim is not None and len(shape) != ndim:
        raise ValueError(f'Expected a {ndim}-dimensional input')
    if not shape or any(d <= 0 for d in shape):
        raise ValueError('Input dimensions must be nonempty')
    if tl.get_backend() not in ('numpy', 'pytorch'):
        raise NotImplementedError('Decompositions support numpy and pytorch backends')
    dtype = str(tl.context(x).get('dtype', '')).removeprefix('torch.')
    if dtype not in ('float32', 'float64', 'complex64', 'complex128'):
        raise TypeError('Input dtype must be float32, float64, complex64, or complex128')
    if not np.isfinite(tl.to_numpy(x)).all():
        raise ValueError('Input must contain only finite values')


def _normalize_rank(rank, dimension):
    if isinstance(rank, bool):
        raise TypeError('Boolean rank is not supported')
    if isinstance(rank, numbers.Integral):
        if rank <= 0:
            raise ValueError('Integer rank must be positive')
        return min(int(rank), dimension)
    if isinstance(rank, numbers.Real):
        if not math.isfinite(rank) or not 0 < rank <= 1:
            raise ValueError('Fractional rank must lie in (0, 1]')
        return max(1, int(rank * dimension))
    raise TypeError('Rank must be a positive integer or a fraction in (0, 1]')


def _need_t(f):
    """Orient a matrix using adjoints; preserve the middle singular vector."""
    @wraps(f)
    def wrapper(self, matrix, *args, **kwargs):
        transposed = tl.shape(matrix)[0] >= tl.shape(matrix)[1]
        factors = f(self, _adjoint(matrix) if transposed else matrix, *args, **kwargs)
        if not transposed:
            return factors
        return tuple(_adjoint(t) if tl.ndim(t) == 2 else t for t in reversed(factors))
    return wrapper


def _conditioning(f):
    """Factor XC and undo C on the final factor (weighted approximation).

    Only invertible square C or a nonzero diagonal vector is accepted. The
    singular values describe XC, and the corrected right factor need not be
    orthonormal. At reduced rank the objective is ||(X-Xhat)C||_F, rather than
    an optimal SVD of X. At full rank reconstruction equals X up to roundoff.
    """
    @wraps(f)
    def conditioned(self, matrix, rank=None, conditioner=None, *args, **kwargs):
        c = self._conditioner if conditioner is None else conditioner
        _validate_tensor(matrix, 2)
        if c is None:
            return f(self, matrix, rank, *args, **kwargs)
        n = tl.shape(matrix)[1]
        if not np.isfinite(tl.to_numpy(c)).all():
            raise ValueError('conditioner must be finite')
        if tl.ndim(c) == 1:
            if tl.shape(c) != (n,) or np.any(tl.to_numpy(c) == 0):
                raise ValueError('Diagonal conditioner must have n nonzero entries')
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                inverse_values = 1 / np.asarray(tl.to_numpy(c), dtype=np.asarray(tl.to_numpy(matrix)).dtype)
            if not np.isfinite(inverse_values).all():
                raise ValueError('Diagonal conditioner inverse is not representable in the matrix dtype')
            weighted = matrix * c
            *factors, right = f(self, weighted, rank, *args, **kwargs)
            corrected = right / c
            if not np.isfinite(tl.to_numpy(corrected)).all():
                raise ArithmeticError('Conditioned factor cannot be represented in the matrix dtype')
            return (*factors, corrected)
        if tl.ndim(c) != 2 or tl.shape(c) != (n, n):
            raise ValueError('conditioner must be square (n,n), or diagonal (n,)')
        try:
            c_context = tl.context(c)
        except (TypeError, AttributeError) as exc:
            raise TypeError('conditioner must use the same backend as the matrix') from exc
        if c_context.get('dtype') != tl.context(matrix).get('dtype'):
            raise TypeError('Square conditioner must match matrix dtype')
        if c_context.get('device') != tl.context(matrix).get('device'):
            raise ValueError('Square conditioner must match matrix device')
        values = np.linalg.svd(tl.to_numpy(c), compute_uv=False)
        real_dtype = np.asarray(tl.to_numpy(c)).real.dtype
        if real_dtype.kind != 'f':
            real_dtype = np.dtype('float64')
        if values[-1] <= np.finfo(real_dtype).eps * n * values[0]:
            raise ValueError('conditioner must be numerically invertible')
        inverse = tl.solve(c, tl.eye(n, **tl.context(c)))
        if not np.isfinite(tl.to_numpy(inverse)).all():
            raise ValueError('Square conditioner inverse is not representable in the matrix dtype')
        *factors, right = f(self, tl.matmul(matrix, c), rank, *args, **kwargs)
        corrected = tl.matmul(right, inverse)
        if not np.isfinite(tl.to_numpy(corrected)).all():
            raise ArithmeticError('Conditioned factor cannot be represented in the matrix dtype')
        return (*factors, corrected)
    return conditioned


class AbstractDecomposer(ABC):
    def __init__(self, rank=None, random_init=ProjectorGenerator.normal, random_state=None):
        self.rank = rank
        self.random_init = random_init
        self.random_state = normalize_random_state(random_state)
        self._conditioner = None

    @abstractmethod
    def _get_rank(self, tensor, rank):
        pass

    @abstractmethod
    def _decompose(self, tensor, rank, **kwargs):
        pass

    def _decompose_big(self, tensor, rank, **kwargs):
        return self._decompose(tensor, rank, **kwargs)

    @abstractmethod
    def compose(self, *factors, **kwargs):
        pass

    def decompose(self, tensor, rank=None, **kwargs):
        rank = self._get_rank(tensor, rank)
        if self._is_big(tensor):
            return self._decompose_big(tensor, rank, **kwargs)
        return self._decompose(tensor, rank, **kwargs)

    def _is_big(self, tensor):
        return sum(tl.shape(tensor)) > DIM_SUM_LIM or any(d > DIM_LIM for d in tl.shape(tensor))

    def set_conditioner(self, conditioner):
        self._conditioner = conditioner

    def get_approximation_error(self, tensor, *factors, relative=True):
        error = tl.norm(tensor - self.compose(*factors), order=2)
        if relative:
            norm = tl.norm(tensor, order=2)
            return error / norm if float(norm) > 0 else error
        return error


class Decomposer(AbstractDecomposer):
    def __init__(self, rank=None, distortion_factor=0.6, random_init=ProjectorGenerator.normal, random_state=None):
        super().__init__(rank, random_init, random_state)
        if not isinstance(distortion_factor, numbers.Real) or not math.isfinite(distortion_factor) or not 0 < distortion_factor <= 1:
            raise ValueError('distortion_factor must lie in (0,1]')
        self.distortion_factor = distortion_factor

    def estimate_stable_rank(self, matrix):
        """Historical sample-count heuristic; does not guarantee error tolerance."""
        n = max(tl.shape(matrix))
        e = self.distortion_factor
        count = int(4 * math.log(n) / (e**2 / 2 - e**3 / 3))
        return max(1, min(count, *tl.shape(matrix)))

    def _get_rank(self, tensor, rank):
        _validate_tensor(tensor, 2)
        selected = self.rank if rank is None else rank
        return self.estimate_stable_rank(tensor) if selected is None else _normalize_rank(selected, min(tl.shape(tensor)))

    @_conditioning
    def decompose(self, tensor, rank=None, **kwargs):
        if 'n_eigenvecs' in kwargs:
            if rank is not None:
                raise ValueError('Specify either rank or n_eigenvecs')
            rank = kwargs.pop('n_eigenvecs')
        return super().decompose(tensor, rank, **kwargs)

    def compose(self, *factors, **kwargs):
        if len(factors) == 2:
            return tl.matmul(*factors)
        if len(factors) == 3:
            left, values, right = factors
            return tl.matmul(left * values, right)
        raise ValueError('Expected two factors or (U,S,Vh)')


class BaseSketch(Decomposer):
    """Column sketches and their minimum-norm least-squares decompositions.

    A size is a positive integer or a fraction of min(matrix.shape), capped
    by that dimension. The default uses compression_ratio. sketch()/__call__
    return selected columns C; decompose() returns (C, pinv(C) @ matrix).
    A conditioner follows Decomposer's weighted approximation contract.
    Explicit random_state follows the other decomposers' local stream rules.
    """
    def __init__(self, sketch_size=None, compression_ratio=0.5, random_state=None):
        super().__init__(rank=sketch_size, random_state=random_state)
        if (isinstance(compression_ratio, bool)
                or not isinstance(compression_ratio, numbers.Real)
                or not math.isfinite(compression_ratio)
                or not 0 < compression_ratio < 1):
            raise ValueError('compression_ratio must lie in (0, 1)')
        self.sketch_size = sketch_size
        self.compression_ratio = compression_ratio
        self.column_indices = None

    def _get_rank(self, tensor, rank):
        _validate_tensor(tensor, 2)
        selected = self.sketch_size if rank is None else rank
        dimension = min(tl.shape(tensor))
        if selected is None:
            selected = self.compression_ratio
        return _normalize_rank(selected, dimension)

    def sketch(self, matrix, sketch_size=None, *, random_state=None):
        size = self._get_rank(matrix, sketch_size)
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        return self._sketch(matrix, size, random_state=rng)

    def __call__(self, matrix, sketch_size=None, *, random_state=None):
        return self.sketch(matrix, sketch_size, random_state=random_state)

    @abstractmethod
    def _sketch(self, matrix, sketch_size, *, random_state):
        pass

    def _decompose(self, matrix, rank, *, random_state=None):
        from tdecomp.utils import pseudo_inverse
        columns = self.sketch(matrix, rank, random_state=random_state)
        return columns, tl.matmul(pseudo_inverse(columns), matrix)


class TensorDecomposer(AbstractDecomposer):
    def _get_rank(self, tensor, rank):
        _validate_tensor(tensor)
        selected = self.rank if rank is None else rank
        shape = tl.shape(tensor)
        capacities = [min(d, math.prod(shape[:i] + shape[i+1:])) for i, d in enumerate(shape)]
        if selected is None:
            return capacities
        if isinstance(selected, (numbers.Real, bool)):
            return [min(_normalize_rank(selected, d), cap) for d, cap in zip(shape, capacities)]
        if not isinstance(selected, (list, tuple, np.ndarray)):
            raise TypeError('Tensor rank must be an integer, fraction, or sequence')
        if len(selected) != len(shape):
            raise ValueError('Rank sequence length must equal tensor dimensionality')
        return [min(_normalize_rank(r, d), cap) for r, d, cap in zip(selected, shape, capacities)]

    def compose(self, core, *factors):
        if len(factors) == 1 and isinstance(factors[0], (list, tuple)):
            factors = factors[0]
        for mode, factor in enumerate(factors):
            core = tl.tenalg.mode_dot(core, factor, mode)
        return core

    def get_approximation_error(self, tensor, *approximation, relative=True):
        core, factors = approximation
        return super().get_approximation_error(tensor, core, *factors, relative=relative)
