"""Heuristic column selection, with column-sized weights and local RNG.

These preserve the sampling interfaces from PR #21, not the approximation
guarantees of a published adaptive-sampling or ColumnSelect algorithm.
"""
import math
import numbers

import numpy as np
import tensorly as tl

from tdecomp._base import BaseSketch, _adjoint
from tdecomp.matrix.importance_generators import fro_norm
from tdecomp.matrix.random_projections import normal
from tdecomp.utils import index_tensor, multinomial

__all__ = ['AdaptiveSamplingSketch', 'ColumnSelectSketch', 'adaptive_sampling',
           'column_select', 'SAMPLING_TECHNIQUES']


def _power_count(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError('power_iterations must be a nonnegative integer')
    return int(value)


def _scaled(matrix):
    maximum = float(tl.max(tl.abs(matrix)))
    return matrix / maximum if maximum else matrix


def _sample_columns(matrix, weights, size, rng):
    """Prefer positive scores, then fill the budget from zero-score columns."""
    values = np.asarray(tl.to_numpy(weights))
    positive_count = int(np.count_nonzero(values > 0))
    if positive_count == 0:
        selected = rng.permutation(tl.shape(matrix)[1])[:size]
    else:
        selected = tl.to_numpy(multinomial(weights, min(size, positive_count), random_state=rng))
        if positive_count < size:
            extra = rng.permutation(np.flatnonzero(values == 0))[:size - positive_count]
            selected = np.concatenate((selected, extra))
    return index_tensor(selected, tl.context(matrix))


class AdaptiveSamplingSketch(BaseSketch):
    """One-shot sampling from a randomized dominant right-vector estimate.

    Initialize v = A.H @ Gaussian(m, 1), then apply power_iterations scaled
    A.H/A passes and sample distinct columns with weights |v_j|^2. A zero
    estimate falls back to squared column norms. This historical class name
    does not imply residual resampling or an error guarantee.
    """
    def __init__(self, sketch_size=None, compression_ratio=0.5,
                 power_iterations=2, random_state=None):
        super().__init__(sketch_size, compression_ratio, random_state)
        self.power_iterations = _power_count(power_iterations)

    def _estimate_column_importance(self, matrix, *, random_state=None):
        rng = self.random_state if random_state is None else random_state
        working = _scaled(matrix)
        vector = tl.matmul(_adjoint(working), normal(tl.shape(matrix)[0], 1,
                                                    context=tl.context(matrix), random_state=rng))
        for _ in range(self.power_iterations):
            maximum = float(tl.max(tl.abs(vector)))
            if maximum == 0:
                break
            vector = tl.matmul(_adjoint(working), tl.matmul(working, vector / maximum))
        maximum = float(tl.max(tl.abs(vector)))
        if maximum == 0:
            return fro_norm(matrix)[0]
        weights = tl.reshape(tl.abs(vector / maximum) ** 2, (-1,))
        return weights / tl.sum(weights)

    def _sketch(self, matrix, sketch_size, *, random_state):
        weights = self._estimate_column_importance(matrix, random_state=random_state)
        self.column_indices = _sample_columns(matrix, weights, sketch_size, random_state)
        return matrix[:, self.column_indices]


class ColumnSelectSketch(BaseSketch):
    """Sample distinct columns from a randomized right-subspace estimate.

    A.H @ Gaussian(m, k) estimates the right subspace, where k=min(10,size).
    Thin SVD removes null directions; stabilized A.H/A power passes refine
    the basis. Column weights are squared row norms of that n-by-k basis.
    epsilon is retained and validated for source compatibility; it does not
    change the requested sample budget or provide an error guarantee.
    """
    def __init__(self, sketch_size=None, compression_ratio=0.5, epsilon=0.1,
                 power_iterations=2, random_state=None):
        super().__init__(sketch_size, compression_ratio, random_state)
        if (isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real)
                or not math.isfinite(epsilon) or not 0 < epsilon < 1):
            raise ValueError('epsilon must lie in (0, 1)')
        self.epsilon = epsilon
        self.power_iterations = _power_count(power_iterations)

    def _fast_leverage_scores(self, matrix, k, *, random_state=None):
        rng = self.random_state if random_state is None else random_state
        working = _scaled(matrix)
        projected = tl.matmul(_adjoint(working), normal(tl.shape(matrix)[0], k,
                                                       context=tl.context(matrix), random_state=rng))
        for iteration in range(self.power_iterations + 1):
            basis, singular, _ = tl.truncated_svd(projected, n_eigenvecs=min(tl.shape(projected)))
            values = tl.to_numpy(singular)
            threshold = np.finfo(values.dtype).eps * max(tl.shape(projected)) * values[0]
            width = int(np.count_nonzero(values > threshold))
            if width == 0:
                return fro_norm(matrix)[0]
            basis = basis[:, :width]
            if iteration < self.power_iterations:
                projected = tl.matmul(_adjoint(working), tl.matmul(working, basis))
        weights = tl.sum(tl.abs(basis) ** 2, axis=1)
        return weights / tl.sum(weights)

    def _sketch(self, matrix, sketch_size, *, random_state):
        weights = self._fast_leverage_scores(matrix, min(10, sketch_size), random_state=random_state)
        self.column_indices = _sample_columns(matrix, weights, sketch_size, random_state)
        return matrix[:, self.column_indices]


def adaptive_sampling(matrix, sketch_size=None, **kwargs):
    """Return columns chosen by AdaptiveSamplingSketch."""
    return AdaptiveSamplingSketch(sketch_size=sketch_size, **kwargs)(matrix)


def column_select(matrix, sketch_size=None, **kwargs):
    """Return columns chosen by ColumnSelectSketch."""
    return ColumnSelectSketch(sketch_size=sketch_size, **kwargs)(matrix)


SAMPLING_TECHNIQUES = {'adaptive_sampling': AdaptiveSamplingSketch,
                       'column_select': ColumnSelectSketch}
