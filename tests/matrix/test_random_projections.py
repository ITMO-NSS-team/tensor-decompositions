import copy
import math
import numpy as np
import pytest
import torch
import tensorly as tl
from tdecomp.matrix.random_projections import *
from tdecomp.matrix.random_projections import RANDOM_GENS
from tdecomp.matrix.decomposer import RandomizedSVD, TwoSidedRandomSVD


def test_ortho_columns_and_rows():
    for shape in [(5,3),(3,5)]:
        q=ortho(*shape,random_state=4)
        product=q.mH@q if shape[0]>=shape[1] else q@q.mH
        assert torch.allclose(product,torch.eye(min(shape),dtype=q.dtype),atol=1e-12)


def test_normal_known_draw_and_normalization():
    expected=np.random.default_rng(5).standard_normal((8,4))/2
    actual=normal(8,4,random_state=5)
    assert np.array_equal(tl.to_numpy(actual),expected)


def test_sparse_iid_values_and_symmetry():
    p=sparse_iid_entries(1000,20,s=3,random_state=7)
    values=tl.to_numpy(p)
    assert set(np.unique(values))=={-math.sqrt(3/20),0,math.sqrt(3/20)}
    assert abs(np.count_nonzero(values>0)-np.count_nonzero(values<0))<300


def test_sparse_jl_exact_distinct_support_and_row_energy():
    p=sparse_jl_matrix(8,4,s=3,random_state=4)
    assert torch.equal(torch.count_nonzero(p,dim=1),torch.full((8,),3,dtype=torch.long))
    assert torch.allclose((p*p).sum(1),torch.ones(8,dtype=p.dtype),atol=1e-14)
    assert set(tl.to_numpy(p).ravel()) <= {-1/math.sqrt(3),0,1/math.sqrt(3)}


def test_four_wise_matches_independent_scalar_field_reference():
    coeff=np.random.default_rng(7).integers(0,256,4)
    def multiply(a,b):
        result=0
        for _ in range(8):
            if b & 1: result ^= a
            a=(a<<1) ^ (0x11b if a&128 else 0)
            b >>= 1
        return result
    expected=[]
    for x in range(32):
        value=int(coeff[3])
        for c in coeff[2::-1]: value=multiply(value,x)^int(c)
        expected.append((2*(value&1)-1)/2)
    assert np.array_equal(tl.to_numpy(four_wise_independent_matrix(8,4,random_state=7)),np.array(expected).reshape(8,4))


def test_explicit_unsupported_structured_generators():
    with pytest.raises(NotImplementedError): lean_walsh(8,4)
    with pytest.raises(ValueError): identity_copies(8,4)
    with pytest.raises(ValueError): four_wise_independent_matrix(100,4)
    assert 'lean_walsh' not in RANDOM_GENS
    p=identity_copies(3,6,random_state=6)
    assert torch.allclose(p@p.T,torch.eye(3,dtype=p.dtype),atol=1e-14)


@pytest.mark.parametrize('cls',[RandomizedSVD,TwoSidedRandomSVD])
@pytest.mark.parametrize('generator',[ProjectorGenerator.normal,ProjectorGenerator.ortho,ProjectorGenerator.sparse_iid_entries,ProjectorGenerator.sparse_jl_matrix,ProjectorGenerator.four_wise_independent_matrix])
def test_supported_sketches_public_svd_paths(cls,generator):
    x=torch.arange(1.,33.,dtype=torch.float64).reshape(8,4)
    dec=cls(rank=2,random_state=3,random_init=generator)
    factors=dec.decompose(x,**({'s':2} if generator==ProjectorGenerator.sparse_jl_matrix else {}))
    assert [tuple(a.shape) for a in factors]==[(8,2),(2,),(2,4)]
    assert all(torch.isfinite(a).all() for a in factors)


def test_projector_context_cache_and_renewal():
    p=Projector(ProjectorGenerator.normal,random_state=5)
    x=torch.ones(3,2,dtype=torch.float64)
    assert p.lproject(x,2).shape==(2,2)
    cached=p.P.clone()
    p.lproject(x,2,renew=False)
    assert torch.equal(p.P,cached)
    p.lproject(x,2,renew=True)
    assert not torch.equal(p.P,cached)
    p.rproject(x,2,renew=False)
    assert p.P.shape==(2,2)
    p.rproject(x.float(),2,renew=False)
    assert p.P.dtype==torch.float32
    old=p.P.clone()
    with pytest.raises(ValueError): p.project(x,2,side='unknown')
    assert torch.equal(p.P,old)


def test_local_rng_state_restore_and_global_isolation():
    rng=np.random.default_rng(2)
    np_before=np.random.get_state()
    torch_before=torch.random.get_rng_state().clone()
    state=copy.deepcopy(rng.bit_generator.state)
    first=normal(8,4,random_state=rng)
    rng.bit_generator.state=state
    second=normal(8,4,random_state=rng)
    assert torch.equal(first,second)
    np_after=np.random.get_state()
    assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]
    assert torch.equal(torch_before,torch.random.get_rng_state())


def test_complex_normal_and_haar_orthogonal_basis():
    context={'dtype':torch.complex128,'device':torch.device('cpu')}
    p=normal(8,4,context=context,random_state=4)
    assert p.is_complex() and torch.count_nonzero(p.imag)>0
    q=ortho(8,4,context=context,random_state=4)
    assert torch.allclose(q.mH@q,torch.eye(4,dtype=q.dtype),atol=1e-12)
