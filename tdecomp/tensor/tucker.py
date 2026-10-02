"""Random projection, sequential SVD, sampling and HOOI Tucker methods."""
import numbers
import numpy as np
import tensorly as tl
from tdecomp._base import TensorDecomposer, _adjoint
from tdecomp._random import normalize_random_state
from tdecomp.matrix.decomposer import RandomizedSVD, _nonnegative_integer
from tdecomp.matrix.random_projections import ProjectorGenerator, normal
from tdecomp.matrix.importance_generators import fro_norm, ridge_leverage

__all__ = ['RPHOSVDDecomposition', 'RSTHOSVDDecomposition', 'RSTDecomposition', 'HOOIDecomposition']


def _basis(matrix, rank, rsvd, rng):
    # Extra null-space vectors are needed when a requested modal rank exceeds
    # the unfolding width. They preserve the declared factor/core shapes.
    width = min(rank, min(tl.shape(matrix)))
    u, _, _ = rsvd.decompose(matrix, width, random_state=rng)
    if tl.shape(u)[1] < rank:
        u, _ = tl.qr(u, mode='complete')
    return u[:, :rank]


class RPHOSVDDecomposition(TensorDecomposer):
    """Independent modal range finders on the original tensor.

    power counts stabilized A.H/A passes; factors are (mode_size,mode_rank).
    """
    def __init__(self, *, rank=None, distortion_factor=0.6, power=3,
                 random_init=ProjectorGenerator.normal, random_state=None):
        super().__init__(rank, random_init, random_state)
        self.power = _nonnegative_integer(power, 'power')
        self.rsvd = RandomizedSVD(power=power, distortion_factor=distortion_factor, random_init=random_init)

    def _decompose(self, tensor, rank, random_state=None, **kwargs):
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        factors = [_basis(tl.unfold(tensor, mode), r, self.rsvd, rng) for mode, r in enumerate(rank)]
        core = tl.tenalg.multi_mode_dot(tensor, factors, transpose=True)
        return core, factors


class RSTHOSVDDecomposition(TensorDecomposer):
    """Sequentially truncated HOSVD: each SVD unfolds the current core."""
    def __init__(self, *, rank=None, oversampling=10, power_iteration=2, distortion_factor=0.1,
                 random_init=ProjectorGenerator.normal, random_state=None):
        super().__init__(rank, random_init, random_state)
        self.oversampling = _nonnegative_integer(oversampling, 'oversampling')
        self.power_iteration = _nonnegative_integer(power_iteration, 'power_iteration')
        self.rsvd = RandomizedSVD(power=power_iteration, oversampling=oversampling,
                                  distortion_factor=distortion_factor, random_init=random_init)

    def _decompose(self, tensor, rank, random_state=None, **kwargs):
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        core, factors = tensor, []
        unfolding_shapes = []
        for mode, r in enumerate(rank):
            unfolded = tl.unfold(core, mode)
            unfolding_shapes.append(tl.shape(unfolded))
            u = _basis(unfolded, r, self.rsvd, rng)
            factors.append(u)
            core = tl.tenalg.mode_dot(core, _adjoint(u), mode)
        self.diagnostics = {'unfolding_shapes': unfolding_shapes, 'oversampling': self.oversampling,
                            'power_iteration': self.power_iteration}
        return core, factors


class RSTDecomposition(TensorDecomposer):
    """Sampled-column Tucker with thresholded Moore-Penrose inverses.

    Modes sample original unfoldings. norm_based uses squared column norms;
    leverage_score uses ridge leverage probabilities. If fewer positive
    columns exist than requested, all positives are kept and remaining zero
    columns are filled uniformly. Diagnostics retain indices/distributions.
    """
    def __init__(self, *, rank=None, sampling_method='norm_based', distortion_factor=0.6,
                 random_init=ProjectorGenerator.normal, random_state=None):
        super().__init__(rank, random_init, random_state)
        if sampling_method not in ('uniform', 'norm_based', 'leverage_score'):
            raise ValueError('sampling_method must be uniform, norm_based, or leverage_score')
        self.sampling_method = sampling_method
        self.sample_indices = []

    def _sample(self, matrix, n, rng):
        from tdecomp.utils import randperm, multinomial
        width = tl.shape(matrix)[1]
        count = min(n, width)
        if self.sampling_method == 'uniform':
            indices = randperm(width, tl.context(matrix), random_state=rng)[:count]
            probs = None
        else:
            probs, _ = (fro_norm(matrix) if self.sampling_method == 'norm_based' else ridge_leverage(matrix))
            positive = int(np.count_nonzero(tl.to_numpy(probs)))
            selected = multinomial(probs, min(count, positive), random_state=rng)
            selected_np = np.asarray(tl.to_numpy(selected), dtype=np.int64)
            if positive < count:
                remaining = np.setdiff1d(np.arange(width), selected_np)
                selected_np = np.concatenate((selected_np, rng.choice(remaining, count-positive, replace=False)))
            # argsort produces backend-native integer indices without coercing
            # complex/float data context to an index dtype.
            indices = selected
            if positive < count:
                from tdecomp.utils import index_tensor
                indices = index_tensor(selected_np, tl.context(matrix))
        return matrix[:, indices], indices, probs

    def _decompose(self, tensor, rank, random_state=None, **kwargs):
        from tdecomp.utils import pseudo_inverse
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        core, factors, indices, probabilities = tensor, [], [], []
        for mode, r in enumerate(rank):
            q, chosen, probs = self._sample(tl.unfold(tensor, mode), r, rng)
            # R-ST can only sample as many columns as the unfolding contains.
            factors.append(q)
            indices.append(chosen)
            probabilities.append(probs)
            core = tl.tenalg.mode_dot(core, pseudo_inverse(q), mode)
        self.sample_indices = indices
        self.diagnostics = {'sampling_method': self.sampling_method, 'indices': indices, 'probabilities': probabilities}
        return core, factors


class HOOIDecomposition(TensorDecomposer):
    """TensorLy 0.9 HOOI, with explicit API mapping and local initialization."""
    def __init__(self, rank=None, init='svd', n_iter_max=100, svd_type='truncated_svd', random_state=None):
        super().__init__(rank, random_state=random_state)
        self.init = init
        self.n_iter_max = _nonnegative_integer(n_iter_max, 'n_iter_max')
        if self.n_iter_max == 0:
            raise ValueError('n_iter_max must be positive')
        self.svd_type = svd_type

    def decompose(self, tensor, rank=None, init=None, n_iter_max=None, svd_type=None, **kwargs):
        count = self.n_iter_max if n_iter_max is None else _nonnegative_integer(n_iter_max, 'n_iter_max')
        if count == 0:
            raise ValueError('n_iter_max must be positive')
        return super().decompose(tensor, rank, init=self.init if init is None else init,
                                 n_iter_max=count, svd=self.svd_type if svd_type is None else svd_type, **kwargs)

    def _decompose(self, tensor, rank, init='svd', random_state=None, **kwargs):
        rng = self.random_state if random_state is None else normalize_random_state(random_state)
        if isinstance(init, str) and init == 'random':
            factors = [normal(d, r, context=tl.context(tensor), random_state=rng) for d, r in zip(tl.shape(tensor), rank)]
            core = tl.tensor(rng.standard_normal(rank), **tl.context(tensor))
            init = (core, factors)
        # TensorLy's randomized_svd accepts a RandomState rather than a modern
        # Generator. Forward a local derived seed without consuming globals.
        from tdecomp._random import rng_integers
        seed = int(rng_integers(rng, 0, 2**31 - 1))
        core, factors = tl.decomposition.tucker(tensor, rank=rank, init=init, random_state=seed, **kwargs)
        # A custom SVD may supply only the unfolding's effective number of
        # columns. Complete an orthogonal basis and zero-pad the core so the
        # normalized modal shape remains explicit without changing the fit.
        if tuple(tl.shape(core)) != tuple(rank):
            old_shape = tl.shape(core)
            padded = tl.zeros(tuple(rank), **tl.context(core))
            padded = tl.index_update(padded, tl.index[tuple(slice(0, d) for d in old_shape)], core)
            core = padded
            completed = []
            for factor, r in zip(factors, rank):
                old_width = tl.shape(factor)[1]
                if old_width < r:
                    q, _ = tl.qr(factor, mode='complete')
                    factor = tl.concatenate((factor, q[:, old_width:r]), axis=1)
                completed.append(factor)
            factors = completed
        return core, factors


DECOMPOSERS = {name: globals()[name] for name in __all__}
