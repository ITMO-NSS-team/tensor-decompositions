"""Column sampling contracts checked against independent NumPy references."""
import copy

import numpy as np
import pytest
import tensorly as tl
import torch

from tdecomp.matrix.sampling_techniques import (
    AdaptiveSamplingSketch, ColumnSelectSketch, adaptive_sampling, column_select,
)

METHODS = [AdaptiveSamplingSketch, ColumnSelectSketch]


@pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('shape', [(4, 11), (11, 4), (1025, 3)])
def test_rectangular_sketch_is_distinct_input_columns_and_lstsq(method, backend, shape):
    values = np.random.default_rng(17).normal(size=shape)
    with tl.backend_context(backend):
        matrix = tl.tensor(values)
        sampler = method(sketch_size=3, random_state=8)
        columns, coefficients = sampler.decompose(matrix)
        indices = tl.to_numpy(sampler.column_indices)
        assert indices.dtype.kind in 'iu' and len(set(indices)) == 3
        assert np.all(indices < shape[1])
        np.testing.assert_array_equal(tl.to_numpy(columns), values[:, indices])
        reference = np.linalg.lstsq(values[:, indices], values, rcond=None)[0]
        np.testing.assert_allclose(tl.to_numpy(coefficients), reference, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(tl.to_numpy(matrix), values)


@pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('power', [0, 2])
def test_rank_one_complex_weights_are_column_scores(method, backend, power):
    left = np.array([1 + 1j, 2, -1j])
    right = np.array([0, 1j, 2, 0, -3j, 1])
    values = np.outer(left, right)
    expected = abs(right) ** 2 / np.sum(abs(right) ** 2)
    with tl.backend_context(backend):
        matrix = tl.tensor(values)
        sampler = method(sketch_size=1, power_iterations=power, random_state=4)
        if method is AdaptiveSamplingSketch:
            scores = sampler._estimate_column_importance(matrix)
        else:
            scores = sampler._fast_leverage_scores(matrix, 3)
        assert tuple(tl.shape(scores)) == (values.shape[1],)
        np.testing.assert_allclose(tl.to_numpy(scores), expected, atol=1e-12, rtol=1e-12)
        factors = sampler.decompose(matrix)
        np.testing.assert_allclose(tl.to_numpy(sampler.compose(*factors)), values, atol=1e-11)


@pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('zero', [False, True])
def test_zero_or_sparse_support_fills_budget_without_duplicates(method, backend, zero):
    values = np.zeros((4, 9))
    if not zero:
        values[:, 8] = [1, 2, 3, 4]
    with tl.backend_context(backend):
        matrix = tl.tensor(values)
        sampler = method(sketch_size=3, random_state=7)
        columns, coefficients = sampler.decompose(matrix)
        indices = tl.to_numpy(sampler.column_indices)
        assert len(indices) == len(set(indices)) == 3
        if not zero:
            assert 8 in indices
        assert np.isfinite(tl.to_numpy(coefficients)).all()
        np.testing.assert_allclose(tl.to_numpy(columns) @ tl.to_numpy(coefficients), values, atol=1e-12)
        assert float(sampler.get_approximation_error(matrix, columns, coefficients)) < 1e-12


@pytest.mark.parametrize('method', METHODS)
def test_seed_repeatability_stream_isolation_and_callable(method):
    matrix = torch.arange(1., 46., dtype=torch.float64).reshape(5, 9)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    first, second = method(sketch_size=2, random_state=19), method(sketch_size=2, random_state=19)
    np.testing.assert_array_equal(first(matrix), second.sketch(matrix))
    other = method(sketch_size=2, random_state=11)
    other(matrix)
    np.testing.assert_array_equal(first(matrix), second(matrix))
    state_before = copy.deepcopy(first.random_state.bit_generator.state)
    np.testing.assert_array_equal(first(matrix, random_state=31), first(matrix, random_state=31))
    assert state_before == first.random_state.bit_generator.state
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('size,expected', [(None, 2), (0.5, 2), (20, 4)])
def test_size_config_recomputed_and_explicit_override(method, size, expected):
    sampler = method(sketch_size=size, compression_ratio=0.5, random_state=2)
    assert sampler(torch.ones(4, 9)).shape == (4, expected)
    assert sampler(torch.ones(2, 7)).shape == (2, 1 if size is None or size == 0.5 else 2)
    assert sampler.sketch_size == size
    assert sampler(torch.ones(4, 9), sketch_size=1).shape == (4, 1)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('size', [0, -1, True, 0., 1.1, float('nan'), '2'])
def test_invalid_size_fails_before_consuming_rng(method, size):
    sampler = method(random_state=5)
    before = copy.deepcopy(sampler.random_state.bit_generator.state)
    with pytest.raises((TypeError, ValueError)):
        sampler(torch.ones(4, 7), sketch_size=size)
    assert before == sampler.random_state.bit_generator.state
    assert sampler.column_indices is None


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('values', [torch.zeros(0, 3), torch.ones(2, 3, 4),
                                  torch.ones(3, 2, dtype=torch.int64),
                                  torch.tensor([[float('nan')]])])
def test_invalid_matrix_fails_before_consuming_rng(method, values):
    sampler = method(random_state=5)
    before = copy.deepcopy(sampler.random_state.bit_generator.state)
    with pytest.raises((TypeError, ValueError)):
        sampler(values)
    assert before == sampler.random_state.bit_generator.state


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('options', [{'compression_ratio': 0}, {'compression_ratio': 1},
                                    {'compression_ratio': True}, {'power_iterations': -1},
                                    {'power_iterations': 0.5}, {'power_iterations': True}])
def test_invalid_configuration(method, options):
    with pytest.raises(ValueError):
        method(**options)


@pytest.mark.parametrize('epsilon', [0, 1, True, float('nan')])
def test_column_select_validates_legacy_epsilon(epsilon):
    with pytest.raises(ValueError):
        ColumnSelectSketch(epsilon=epsilon)


@pytest.mark.parametrize('function', [adaptive_sampling, column_select])
def test_functional_sketch_wrappers(function):
    matrix = torch.eye(4, dtype=torch.float64)
    assert function(matrix, sketch_size=2, random_state=3).shape == (4, 2)
    with pytest.raises(TypeError):
        function(matrix, unsupported_keyword=True)
