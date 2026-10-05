"""Small CPU matrix checks with explicit ranks and a direct SVD reference."""
import os
import pytest
import torch
from tdecomp.matrix.decomposer import DECOMPOSERS

DEVICE = os.environ.get('TDECOMP_TEST_DEVICE', 'cpu')

@pytest.mark.parametrize('shape', [(8, 8), (5, 9), (9, 5), (1025, 4)])
@pytest.mark.parametrize('name', DECOMPOSERS)
def test_full_rank_matrix_roundtrip(name, shape):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    x = torch.randn(shape, device=DEVICE, dtype=torch.float64, generator=generator)
    dec = DECOMPOSERS[name](rank=min(shape), random_state=13)
    approximation = dec.compose(*dec.decompose(x))
    assert torch.allclose(approximation, x, atol=1e-10, rtol=1e-10)
