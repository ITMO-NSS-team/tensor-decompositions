"""Small anisotropic CPU Tucker checks; no import-time tensor allocation."""
import os
import pytest
import torch
from tdecomp.tensor.tucker import DECOMPOSERS

DEVICE = os.environ.get('TDECOMP_TEST_DEVICE', 'cpu')

@pytest.mark.parametrize('name', DECOMPOSERS)
def test_full_rank_tensor_roundtrip(name):
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    x = torch.randn((3,4,5), device=DEVICE, dtype=torch.float64, generator=generator)
    dec = DECOMPOSERS[name](random_state=7)
    core, factors = dec.decompose(x)
    reconstruction = dec.compose(core, *factors)
    assert torch.allclose(reconstruction, x, rtol=1e-9, atol=1e-9)
