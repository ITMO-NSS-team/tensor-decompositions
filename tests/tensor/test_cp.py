"""CP component-rank, reconstruction, backend and convergence contracts."""
import numpy as np
import pytest
import tensorly as tl
import torch

from tdecomp.tensor import CPDecomposition


@pytest.fixture(params=['numpy', 'pytorch'])
def backend(request):
    tl.set_backend(request.param)
    return request.param


def rank_one(shape=(3, 4, 5), dtype='float64'):
    factors = [tl.tensor(np.linspace(1, 2, size).astype(dtype)[:, None]) for size in shape]
    return CPDecomposition().compose(tl.ones((1,), **tl.context(factors[0])), factors)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_rank_one_roundtrip_and_native_context(backend, dtype):
    x = rank_one(dtype=dtype)
    if dtype.startswith('complex'):
        x = x * (1 + 2j)
    dec = CPDecomposition(rank=1, random_state=7, normalize_factors=True, tol=1e-5)
    weights, factors = dec.decompose(x)
    assert tl.shape(weights) == (1,)
    assert [tl.shape(factor) for factor in factors] == [(size, 1) for size in tl.shape(x)]
    assert all(tl.context(value) == tl.context(x) for value in (weights, *factors))
    np.testing.assert_allclose(tl.to_numpy(dec.compose(weights, factors)), tl.to_numpy(x), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tl.to_numpy(dec.compose(weights, *factors)), tl.to_numpy(x), rtol=2e-5, atol=2e-5)
    assert float(dec.get_approximation_error(x)) < 2e-5
    assert float(dec.get_approximation_error(x, weights, factors)) < 2e-5
    assert 1 <= dec.n_iterations_ < dec.n_iter_max


def test_component_rank_is_not_a_tucker_modal_rank(backend):
    # A (2,2,2) tensor can have real CP rank 3: components must not be capped
    # by the smallest mode. Use a zero tensor to avoid a nonunique ALS fit.
    dec = CPDecomposition(rank=3)
    weights, factors = dec.decompose(tl.zeros((2, 2, 2)))
    assert tl.shape(weights) == (3,)
    assert [tl.shape(factor) for factor in factors] == [(2, 3)] * 3


def test_large_tensor_route(backend):
    x = rank_one(shape=(1030, 2, 2))
    dec = CPDecomposition(rank=1, random_state=9, n_iter_max=5)
    weights, factors = dec.decompose(x)
    np.testing.assert_allclose(tl.to_numpy(dec.compose(weights, factors)), tl.to_numpy(x), rtol=1e-6, atol=1e-6)


def test_fractional_and_default_rank_recomputed_without_mutating_configuration(backend):
    dec = CPDecomposition(rank=0.5)
    for shape, expected in [((3, 4, 5), 1), ((6, 4, 5), 2)]:
        assert tl.shape(dec.decompose(tl.zeros(shape))[0]) == (expected,)
        assert dec.rank == 0.5
    assert tl.shape(dec.decompose(tl.zeros((3, 4, 5)), rank=2)[0]) == (2,)
    assert dec.rank == 0.5
    assert tl.shape(CPDecomposition().decompose(tl.zeros((3, 4, 5)))[0]) == (3,)


@pytest.mark.parametrize('rank', [2, [2], [2, 2, 2], (2, 2, 2), np.array([2, 2, 2])])
def test_legacy_equal_rank_sequences(backend, rank):
    weights, _ = CPDecomposition(rank=rank).decompose(tl.zeros((3, 4, 5)))
    assert tl.shape(weights) == (2,)


@pytest.mark.parametrize('rank,error', [
    (True, TypeError), (0, ValueError), (-1, ValueError), (0.0, ValueError),
    (1.5, ValueError), (float('nan'), ValueError), ('2', TypeError),
    ([], ValueError), ([2, 3, 2], ValueError), ([2, 2], ValueError),
    ([True], TypeError), ([[2]], ValueError),
])
def test_invalid_rank_rejected(backend, rank, error):
    with pytest.raises(error):
        CPDecomposition(rank=rank).decompose(tl.zeros((3, 4, 5)))


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_input_rejected(backend, value):
    x = tl.tensor(np.full((2, 3, 4), value))
    with pytest.raises(ValueError, match='finite'):
        CPDecomposition(rank=1).decompose(x)


def test_input_boundary(backend):
    with pytest.raises(ValueError, match='at least two'):
        CPDecomposition(rank=1).decompose(tl.ones((3,)))
    with pytest.raises(ValueError, match='nonempty'):
        CPDecomposition(rank=1).decompose(tl.ones((0, 3)))
    with pytest.raises(TypeError, match='dtype'):
        CPDecomposition(rank=1).decompose(tl.tensor(np.ones((2, 3, 4), dtype=np.int64)))


def test_zero_tensor_and_error_semantics(backend):
    x = tl.zeros((3, 4, 5))
    dec = CPDecomposition(rank=2, normalize_factors=True)
    weights, factors = dec.decompose(x)
    assert dec.n_iterations_ == 0
    assert dec.errors_ == []
    assert float(dec.get_approximation_error(x)) == 0
    assert float(dec.get_approximation_error(x, weights, *factors, relative=False)) == 0
    assert all(np.isfinite(tl.to_numpy(factor)).all() for factor in factors)
    # The norm of the difference from zero is used when relative error has
    # a zero denominator, consistently with the existing decomposition API.
    nonzero = [tl.ones(tl.shape(factor)) for factor in factors]
    assert float(dec.get_approximation_error(x, tl.ones((2,)), nonzero)) > 0


def test_cached_error_requires_fit(backend):
    with pytest.raises(RuntimeError, match='Decompose'):
        CPDecomposition(rank=1).get_approximation_error(rank_one())


@pytest.mark.parametrize('seed', [None, 9])
def test_random_stream_is_local_and_integer_seed_reproducible(backend, seed):
    x = tl.tensor(np.random.default_rng(4).normal(size=(3, 4, 5)))
    np.random.seed(77)
    torch.manual_seed(77)
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    first = CPDecomposition(rank=2, random_state=seed, n_iter_max=5, tol=0).decompose(x)
    second = CPDecomposition(rank=2, random_state=seed, n_iter_max=5, tol=0).decompose(x)
    if seed is not None:
        for a, b in zip((first[0], *first[1]), (second[0], *second[1])):
            np.testing.assert_array_equal(tl.to_numpy(a), tl.to_numpy(b))
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())


@pytest.mark.parametrize('linesearch', [False, True])
def test_actual_iteration_count_with_disabled_tolerance(backend, linesearch):
    x = tl.tensor(np.random.default_rng(4).normal(size=(3, 4, 5)))
    dec = CPDecomposition(rank=2, random_state=9, n_iter_max=10, tol=0, linesearch=linesearch)
    dec.decompose(x)
    assert dec.n_iterations_ == 10
    assert 0 < len(dec.errors_) <= 10
    if linesearch:
        assert len(dec.errors_) < dec.n_iterations_


def test_per_call_options_and_seed_override(backend):
    x = rank_one()
    dec = CPDecomposition(rank=2, n_iter_max=3, random_state=5)
    weights, _ = dec.decompose(x, rank=1, n_iter_max=20, tol=1e-10, init='svd', random_state=8)
    assert tl.shape(weights) == (1,)
    assert dec.n_iterations_ < 20
    assert dec.rank == 2 and dec.n_iter_max == 3
    with pytest.raises(TypeError, match='Unsupported CP'):
        dec.decompose(x, unsupported=True)


@pytest.mark.parametrize('options,error', [
    ({'n_iter_max': True}, TypeError), ({'n_iter_max': 0}, ValueError),
    ({'n_iter_max': 1.5}, TypeError), ({'tol': -1}, ValueError),
    ({'tol': np.inf}, ValueError), ({'tol': True}, TypeError),
    ({'init': 'unsupported'}, ValueError), ({'normalize_factors': 1}, TypeError),
    ({'linesearch': 'yes'}, TypeError),
])
def test_invalid_solver_options(options, error):
    with pytest.raises(error):
        CPDecomposition(**options)
    with pytest.raises(error):
        CPDecomposition(rank=1).decompose(rank_one(), **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_device_preserved():
    tl.set_backend('pytorch')
    x = rank_one().cuda()
    dec = CPDecomposition(rank=1, random_state=9)
    weights, factors = dec.decompose(x)
    assert all(value.device == x.device for value in (weights, *factors))
    assert dec.compose(weights, factors).device == x.device
    assert float(dec.get_approximation_error(x)) < 1e-6
