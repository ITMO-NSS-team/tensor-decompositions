"""Input-by-output sketches, with explicit normalization and local RNG.

Normal and sparse sketches preserve squared norms in expectation under right
projection. ortho gives an orthonormal basis, a contraction when reduced.
No universal JL guarantee is claimed for every generator.
"""
from enum import Enum
from functools import partial, partialmethod
import math
import numbers
import numpy as np
import tensorly as tl
from tdecomp._random import normalize_random_state, rng_integers

__all__ = ['normal', 'ortho', 'sparse_iid_entries', 'sparse_jl_matrix',
           'four_wise_independent_matrix', 'lean_walsh', 'identity_copies', 'Projector', 'ProjectorGenerator']


def _dimensions(rows, cols):
    for name, value in (('rows', rows), ('cols', cols)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')


def _tensor(values, context):
    return tl.tensor(values, **dict(context or {}))


def normal(rows, cols, context=None, *, random_state=None):
    """Gaussian entries with E|P_ij|^2=1/cols; circular complex if requested."""
    _dimensions(rows, cols)
    rng = normalize_random_state(random_state)
    values = rng.standard_normal((rows, cols))
    if 'complex' in str((context or {}).get('dtype', '')):
        values = (values + 1j * rng.standard_normal((rows, cols))) / math.sqrt(2)
    return _tensor(values / math.sqrt(cols), context)


def ortho(rows, cols, context=None, *, random_state=None):
    """Haar orthonormal columns when rows >= cols, otherwise rows."""
    _dimensions(rows, cols)
    p = normal(max(rows, cols), min(rows, cols), context=context, random_state=random_state)
    q, r = tl.qr(p, mode='reduced')
    diagonal = tl.diag(r)
    absolute = tl.abs(diagonal)
    safe = tl.where(absolute > 0, absolute, tl.ones(tl.shape(absolute), **tl.context(absolute)))
    phase = tl.where(absolute > 0, diagonal / safe, tl.ones(tl.shape(diagonal), **tl.context(diagonal)))
    q = q * phase
    return q if rows >= cols else tl.conj(tl.transpose(q))


def sparse_iid_entries(d, k, s=3, context=None, *, random_state=None):
    """Achlioptas entries +/-sqrt(s/k), each with probability 1/(2s)."""
    _dimensions(d, k)
    if isinstance(s, bool) or not isinstance(s, numbers.Real) or not math.isfinite(s) or s < 1:
        raise ValueError('s must be finite and >= 1')
    u = normalize_random_state(random_state).random((d, k))
    signs = (u < 1 / (2 * s)).astype(float) - (u >= 1 - 1 / (2 * s)).astype(float)
    return _tensor(signs * math.sqrt(s / k), context)


def sparse_jl_matrix(d, k, s=3, context=None, *, random_state=None):
    """Exactly s distinct output coordinates per input, with +/-1/sqrt(s).

    JL bounds require additional relations among s, k, distortion and vectors.
    """
    _dimensions(d, k)
    if isinstance(s, bool) or not isinstance(s, numbers.Integral) or not 1 <= s <= k:
        raise ValueError('s must be an integer in [1, k]')
    rng = normalize_random_state(random_state)
    result = np.zeros((d, k))
    for row in range(d):
        columns = rng.choice(k, size=s, replace=False)
        result[row, columns] = (2 * rng_integers(rng, 0, 2, s) - 1) / math.sqrt(s)
    return _tensor(result, context)


def four_wise_independent_matrix(d, k, context=None, *, random_state=None):
    """Four-wise signs from a cubic over GF(256), limited to d*k <= 256.

    Four distinct points give independent uniform field values through the
    invertible Vandermonde map. A nonzero binary functional (the low bit)
    gives unbiased signs. Field polynomial: x^8+x^4+x^3+x+1.
    """
    _dimensions(d, k)
    if d * k > 256:
        raise ValueError('four_wise_independent_matrix supports at most 256 entries')
    rng = normalize_random_state(random_state)
    coefficients = rng_integers(rng, 0, 256, 4)
    points = np.arange(d * k, dtype=np.int64)

    def multiply(a, b):
        a = np.asarray(a).copy()
        b = np.broadcast_to(b, a.shape).copy()
        result = np.zeros_like(a)
        for _ in range(8):
            result ^= np.where(b & 1, a, 0)
            a = (a << 1) ^ np.where(a & 128, 0x11B, 0)
            b >>= 1
        return result

    values = np.full(points.shape, coefficients[3], dtype=np.int64)
    for coefficient in coefficients[2::-1]:
        values = multiply(values, points) ^ coefficient
    return _tensor((2 * (values & 1) - 1).reshape(d, k) / math.sqrt(k), context)


def lean_walsh(d, k, context=None, *, random_state=None):
    """Reserved name; the previous matrix was not a Lean Walsh transform."""
    _dimensions(d, k)
    raise NotImplementedError('Lean Walsh is unavailable until its rectangular construction is verified')


def identity_copies(d, k, context=None, *, random_state=None):
    """Permuted I_d copies / sqrt(k/d); an isometric expansion, k multiple d."""
    _dimensions(d, k)
    if k < d or k % d:
        raise ValueError('identity_copies requires k >= d and k divisible by d')
    rng = normalize_random_state(random_state)
    result = np.tile(np.eye(d), (1, k // d))[:, rng.permutation(k)] / math.sqrt(k // d)
    return _tensor(result, context)


class ProjectorGenerator(Enum):
    normal = partial(normal)
    ortho = partial(ortho)
    sparse_iid_entries = partial(sparse_iid_entries)
    sparse_jl_matrix = partial(sparse_jl_matrix)
    four_wise_independent_matrix = partial(four_wise_independent_matrix)
    lean_walsh = partial(lean_walsh)
    identity_copies = partial(identity_copies)


class Projector:
    """Owns cached P (output,input). renew consumes its local NumPy stream.

    Save/restore the stream with its standard NumPy state API to resume draws.
    renew=False reuses P only for an identical shape/context/generator contract.
    """
    def __init__(self, mode, random_state=None):
        self.P = None
        self.mode = mode
        self.random_state = normalize_random_state(random_state)
        self._cache_key = None

    def generate_P(self, d, k, context, **generator_kws):
        rng = generator_kws.pop('random_state', self.random_state)
        return self.mode.value(d, k, context=context, random_state=rng, **generator_kws)

    def project(self, tensor, proj_dim, *, side, renew=True, **gen_kws):
        if side not in ('left', 'right'):
            raise ValueError("side must be 'left' or 'right'")
        if tl.ndim(tensor) != 2:
            raise ValueError('Projector accepts matrices only')
        d = tl.shape(tensor)[0 if side == 'left' else 1]
        _dimensions(d, proj_dim)
        context = tl.context(tensor)
        key = (d, proj_dim, side, tl.get_backend(), str(context), repr(gen_kws), self.mode)
        if self.P is None or renew or key != self._cache_key:
            p = self.generate_P(d, proj_dim, context, **gen_kws)
            self.P = tl.conj(tl.transpose(p))
            self._cache_key = key
        return tl.matmul(self.P, tensor) if side == 'left' else tl.matmul(tensor, tl.conj(tl.transpose(self.P)))

    lproject = partialmethod(project, side='left')
    rproject = partialmethod(project, side='right')


RANDOM_GENS = {name: globals()[name] for name in __all__ if name not in ('Projector', 'ProjectorGenerator', 'lean_walsh')}
