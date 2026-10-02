"""Explicit functional matrix decomposition wrappers."""
from tdecomp.matrix.decomposer import RandomizedSVD, TwoSidedRandomSVD, CURDecomposition

__all__ = ['rsvd', 'r2svd', 'cur']


def rsvd(matrix, n_eigenvecs=None, **kwargs):
    """Randomized SVD; keywords configure RandomizedSVD explicitly."""
    return RandomizedSVD(**kwargs).decompose(matrix, rank=n_eigenvecs)


def r2svd(matrix, n_eigenvecs=None, **kwargs):
    """Two-sided SVD; keywords configure TwoSidedRandomSVD explicitly."""
    return TwoSidedRandomSVD(**kwargs).decompose(matrix, rank=n_eigenvecs)


def cur(matrix, n_eigenvecs=None, **kwargs):
    """Top-k CUR; keywords configure CURDecomposition explicitly."""
    return CURDecomposition(**kwargs).decompose(matrix, rank=n_eigenvecs)
