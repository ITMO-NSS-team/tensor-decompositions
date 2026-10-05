import pytest
import torch
import tensorly as tl
from tdecomp._base import _need_t
from tdecomp.matrix.decomposer import SVDDecomposition


def test_need_t_complex_svd_roundtrip():
    @_need_t
    def factors(self, matrix):
        return tl.truncated_svd(matrix, n_eigenvecs=min(matrix.shape))
    matrix = torch.tensor([[1+1j, 2-1j], [3j, 1], [2, -1j]], dtype=torch.complex128)
    u,s,vh = factors(None,matrix)
    assert torch.allclose((u*s)@vh,matrix,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('conditioner', [torch.tensor([1.,100.],dtype=torch.float64), torch.diag(torch.tensor([1.,100.],dtype=torch.float64))])
def test_conditioner_weighted_spectrum_and_full_reconstruction(conditioner):
    matrix=torch.eye(2,dtype=torch.float64)
    dec=SVDDecomposition(rank=2)
    u,s,vh=dec.decompose(matrix,conditioner=conditioner)
    assert torch.allclose(dec.compose(u,s,vh),matrix,atol=1e-12,rtol=1e-12)
    assert torch.allclose(s,torch.tensor([100.,1.],dtype=s.dtype))
    assert not torch.allclose(vh@vh.T,torch.eye(2,dtype=vh.dtype))


@pytest.mark.parametrize('conditioner', [torch.tensor([[1.],[0.]]),torch.tensor([1.,0.]),torch.zeros(2,2),torch.tensor([1.,float('nan')])])
def test_invalid_conditioner_raises_before_result(conditioner):
    with pytest.raises(ValueError):
        SVDDecomposition().decompose(torch.eye(2),conditioner=conditioner)


def test_compose_svd():
    matrix=torch.tensor([[1.,2.],[3.,4.]])
    dec=SVDDecomposition()
    factors=dec.decompose(matrix)
    assert torch.allclose(dec.compose(*factors),matrix,atol=1e-5)
    u,s,vh=factors
    assert torch.allclose(dec.compose(u*s,vh),matrix,atol=1e-5)


def test_square_conditioner_dtype_mismatch_is_early_diagnostic():
    with pytest.raises(TypeError,match='dtype'):
        SVDDecomposition().decompose(torch.eye(2,dtype=torch.float64),conditioner=torch.eye(2,dtype=torch.float32))


def test_tiny_conditioner_unrepresentable_inverse_early_diagnostic():
    with pytest.raises(ValueError,match='representable'):
        SVDDecomposition().decompose(torch.eye(2,dtype=torch.float64),conditioner=torch.tensor([1.,1e-320],dtype=torch.float64))
