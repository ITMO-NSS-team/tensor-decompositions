"""Real nonnegative row/column importance probabilities (NumPy/PyTorch)."""
from enum import Enum
from functools import partial
import math
import numpy as np
import tensorly as tl
from tdecomp.types import TensorLike

__all__ = ['l1_norm', 'l2_norm', 'linf_norm', 'fro_norm', 'ridge_leverage', 'ImportanceComputer']


def _normalize_importances(col_scores, row_scores):
    """Zero total score means uniform selection; finite scores only."""
    result = []
    for scores in (col_scores, row_scores):
        values = tl.to_numpy(scores)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Importance scores must be finite and nonnegative')
        maximum = float(tl.max(scores))
        if maximum == 0:
            result.append(tl.ones(tl.shape(scores), **tl.context(scores)) / tl.shape(scores)[0])
        else:
            scaled = scores / maximum
            result.append(scaled / tl.sum(scaled))
    return tuple(result)


def _norm_importances(matrix, order):
    maximum = float(tl.max(tl.abs(matrix)))
    scaled = matrix / maximum if maximum else matrix
    return _normalize_importances(tl.norm(scaled, order=order, axis=0), tl.norm(scaled, order=order, axis=1))


def l1_norm(matrix):
    return _norm_importances(matrix, 1)


def l2_norm(matrix):
    return _norm_importances(matrix, 2)


def linf_norm(matrix):
    return _norm_importances(matrix, float('inf'))


def fro_norm(matrix):
    """Squared absolute entries; valid for real and complex matrices."""
    scale = float(tl.max(tl.abs(matrix)))
    squared = tl.abs(matrix / scale) ** 2 if scale else tl.abs(matrix) ** 2
    return _normalize_importances(tl.sum(squared, axis=0), tl.sum(squared, axis=1))


def ridge_leverage(matrix, lam=None):
    """Ridge leverage from SVD: sums |U_ij|^2*s_j^2/(s_j^2+lam).

    The analogous expression using V gives column scores. This avoids the
    cancellation of G-G(G+lam I)^-1G. lam is strictly positive; the default
    is 1e-6*||X||_F^2/max(shape), with uniform scores for a zero input.
    """
    from tdecomp._base import _validate_tensor
    _validate_tensor(matrix, 2)
    if not np.isfinite(tl.to_numpy(matrix)).all():
        raise ValueError('matrix must be finite')
    u, s, vh = tl.truncated_svd(matrix, n_eigenvecs=min(tl.shape(matrix)))
    maximum = float(tl.max(s))
    if lam is None:
        if maximum == 0:
            factors = tl.zeros_like(s)
        else:
            squared = (s / maximum) ** 2
            ridge = 1e-6 * float(tl.sum(squared)) / max(tl.shape(matrix))
            factors = squared / (squared + ridge)
    else:
        if isinstance(lam, bool) or not np.isscalar(lam) or not math.isfinite(lam) or lam <= 0:
            raise ValueError('lam must be finite and strictly positive')
        scale = max(maximum, math.sqrt(lam))
        squared = (s / scale) ** 2
        ridge = (math.sqrt(lam) / scale) ** 2
        denominator = squared + ridge
        safe = tl.where(denominator > 0, denominator, tl.ones(tl.shape(s), **tl.context(s)))
        factors = squared / safe
    row_scores = tl.sum(tl.abs(u) ** 2 * factors, axis=1)
    col_scores = tl.sum(tl.abs(vh) ** 2 * tl.reshape(factors, (-1, 1)), axis=0)
    return _normalize_importances(col_scores, row_scores)


class ColumnRowImportancesGenerator(Enum):
    l1_norm = partial(l1_norm)
    l2_norm = partial(l2_norm)
    linf_norm = partial(linf_norm)
    fro_norm = partial(fro_norm)
    ridge_leverage = partial(ridge_leverage)


class ImportanceComputer:
    def __init__(self, mode):
        self.mode = mode

    def compute(self, matrix):
        return self.mode.value(matrix)


IMPORTANCE_GENS = {name: globals()[name] for name in __all__ if name != 'ImportanceComputer'}
