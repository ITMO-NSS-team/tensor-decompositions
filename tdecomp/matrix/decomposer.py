"""Matrix factorizations with stable power iterations and explicit sampling."""
import math
import numbers
import numpy as np
import tensorly as tl
from tdecomp._base import Decomposer, _adjoint
from tdecomp._random import normalize_random_state
from tdecomp.matrix.random_projections import ProjectorGenerator
from tdecomp.matrix.importance_generators import ColumnRowImportancesGenerator

__all__ = ['SVDDecomposition', 'RandomizedSVD', 'TwoSidedRandomSVD', 'CURDecomposition']


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


class SVDDecomposition(Decomposer):
    def _decompose(self, matrix, rank, **kwargs):
        u, s, vh = tl.truncated_svd(matrix, n_eigenvecs=rank)
        return u[:, :rank], s[:rank], vh[:rank, :]


class RandomizedSVD(Decomposer):
    """QR-stabilized randomized range finder followed by a small exact SVD.

    power counts alternating A.H/A passes; power=0 uses A@Omega directly.
    Work is O((2*power+1)*m*n*(rank+oversampling)); no Gram matrix is built.
    distortion_factor tunes an automatic stable-rank heuristic, not a bound
    on approximation error. Small and large input paths use the same method.
    """
    def __init__(self, rank=None, power=3, distortion_factor=0.6,
                 random_init=ProjectorGenerator.normal, random_state=None, oversampling=0):
        super().__init__(rank, distortion_factor, random_init, random_state)
        self.power = _nonnegative_integer(power, 'power')
        self.oversampling = _nonnegative_integer(oversampling, 'oversampling')

    def estimate_stable_rank(self, matrix):
        from tdecomp.utils import svdvals
        singular = svdvals(matrix)
        maximum = float(tl.max(singular))
        if maximum == 0:
            return 1
        squared = (singular / maximum) ** 2
        heuristic = float(tl.sum(squared)) / self.distortion_factor
        return max(1, min(min(tl.shape(matrix)), int(heuristic)))

    def _range(self, matrix, width, rng, **generator_kws):
        omega = self.random_init.value(tl.shape(matrix)[1], width, context=tl.context(matrix), random_state=rng, **generator_kws)
        scale = float(tl.max(tl.abs(matrix)))
        working = matrix / scale if scale else matrix
        q, _ = tl.qr(tl.matmul(working, omega), mode='reduced')
        for _ in range(self.power):
            z, _ = tl.qr(tl.matmul(_adjoint(working), q), mode='reduced')
            q, _ = tl.qr(tl.matmul(working, z), mode='reduced')
        return q

    def _decompose(self, matrix, rank, random_state=None, **generator_kws):
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        width = min(rank + self.oversampling, min(tl.shape(matrix)))
        q = self._range(matrix, width, rng, **generator_kws)
        b = tl.matmul(_adjoint(q), matrix)
        u, s, vh = tl.truncated_svd(b, n_eigenvecs=rank)
        return tl.matmul(q, u[:, :rank]), s[:rank], vh[:rank, :]

    def _decompose_big(self, matrix, rank, **kwargs):
        return self._decompose(matrix, rank, **kwargs)


class TwoSidedRandomSVD(RandomizedSVD):
    """Independent row/column sketches; small SVD of Q1.H@A@Q2.

    This class has no size-dependent switch to a one-sided method. It is a
    projection approximation, with no universal optimality guarantee.
    """
    def __init__(self, rank=None, distortion_factor=0.6, random_init=ProjectorGenerator.normal,
                 random_state=None, oversampling=0, power=0):
        super().__init__(rank, power, distortion_factor, random_init, random_state, oversampling)

    def _decompose(self, matrix, rank, random_state=None, **generator_kws):
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        width = min(rank + self.oversampling, min(tl.shape(matrix)))
        q1 = self._range(matrix, width, rng, **generator_kws)
        q2 = self._range(_adjoint(matrix), width, rng, **generator_kws)
        b = tl.matmul(_adjoint(q1), tl.matmul(matrix, q2))
        u, s, vh = tl.truncated_svd(b, n_eigenvecs=rank)
        return tl.matmul(q1, u[:, :rank]), s[:rank], tl.matmul(vh[:rank, :], _adjoint(q2))


class CURDecomposition(Decomposer):
    """Top-k importance rows/columns with the Moore-Penrose intersection.

    Singular intersections are permitted and yield a finite approximation.
    Exact recovery requires the samples/intersection to preserve matrix rank;
    this deterministic selection is not a globally optimal SVD or a JL map.
    """
    def __init__(self, rank=None, distortion_factor=0.6,
                 random_init=ColumnRowImportancesGenerator.l2_norm, random_state=None):
        super().__init__(rank, distortion_factor, random_init, random_state)
        self.column_indices = None
        self.row_indices = None

    def _decompose(self, matrix, rank, **kwargs):
        from tdecomp.utils import pseudo_inverse
        c, w, r = self.select_rows_cols(matrix, rank)
        return c, pseudo_inverse(w), r

    def _importance(self, matrix):
        return self.random_init.value(matrix)

    def select_rows_cols(self, matrix, rank):
        cols, rows = self._importance(matrix)
        column_indices = tl.sort(tl.argsort(cols, 0)[-rank:], 0)
        row_indices = tl.sort(tl.argsort(rows, 0)[-rank:], 0)
        c, r = matrix[:, column_indices], matrix[row_indices, :]
        w = r[:, column_indices]
        self.column_indices, self.row_indices = column_indices, row_indices
        return c, w, r

    def compose(self, *factors, **kwargs):
        c, u, r = factors
        return tl.matmul(c, tl.matmul(u, r))


DECOMPOSERS = {name: globals()[name] for name in __all__}
