import tensorly as tl
import torch

import tdecomp

def test_pseudo_inverse():
    tl.set_backend("pytorch")
    C = torch.rand((3, 3))
    c_inv_torch = torch.linalg.pinv(C)
    c_inv_solve = tdecomp.utils.pseudo_inverse(C)
    assert torch.allclose(c_inv_solve, c_inv_torch, atol=1e-5)