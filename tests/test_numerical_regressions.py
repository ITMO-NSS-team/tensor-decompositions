"""Audit regressions checked against independent linear algebra references."""
import copy
import numpy as np
import pytest
import torch
import tensorly as tl
from tdecomp.matrix.decomposer import SVDDecomposition, RandomizedSVD, TwoSidedRandomSVD, CURDecomposition
from tdecomp.tensor.tucker import RPHOSVDDecomposition, RSTHOSVDDecomposition, RSTDecomposition, HOOIDecomposition
from tdecomp.matrix.importance_generators import fro_norm,ridge_leverage,l1_norm,l2_norm,linf_norm
from tdecomp.matrix.functional import rsvd,r2svd,cur
from tdecomp.output_formats import ModalDecomposition
from tdecomp.utils import pseudo_inverse,conjugate_gradient,svd_solver_tikhonov,multinomial,randperm,topk_ids


@pytest.mark.parametrize('shape',[(8,5),(5,8)])
@pytest.mark.parametrize('rank',[1,2,1.0])
def test_exact_svd_target_rank_and_best_error(shape,rank):
    x=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
    dec=SVDDecomposition(rank=rank)
    u,s,vh=dec.decompose(x)
    k=rank if isinstance(rank,int) else min(shape)
    assert [tuple(a.shape) for a in (u,s,vh)]==[(shape[0],k),(k,),(k,shape[1])]
    reference=torch.linalg.svdvals(x)
    assert torch.allclose(torch.linalg.norm(x-dec.compose(u,s,vh)),torch.linalg.vector_norm(reference[k:]),atol=1e-12)


@pytest.mark.parametrize('bad',[0,-1,True,False,0.0,-0.1,1.1,float('nan'),float('inf'),'2'])
def test_invalid_rank_early_error(bad):
    dec=RandomizedSVD(random_state=2)
    before=copy.deepcopy(dec.random_state.bit_generator.state)
    with pytest.raises((ValueError,TypeError)): dec.decompose(torch.eye(2),bad)
    assert before==dec.random_state.bit_generator.state


@pytest.mark.parametrize('cls',[RPHOSVDDecomposition,RSTHOSVDDecomposition,RSTDecomposition,HOOIDecomposition])
@pytest.mark.parametrize('rank',[2,0.5,[2,2,2]])
def test_tensor_rank_repeat_calls_new_shape(cls,rank):
    dec=cls(rank=rank,random_state=3)
    for shape in [(3,4,5),(5,3,4)]:
        x=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
        for _ in range(2):
            core,factors=dec.decompose(x)
            expected=dec._get_rank(x,None)
            assert list(core.shape)==expected
            assert [list(q.shape) for q in factors]==[[d,r] for d,r in zip(shape,expected)]
        assert dec.rank==rank


@pytest.mark.parametrize('power',range(4))
@pytest.mark.parametrize('transpose',[False,True])
def test_rank_one_rsvd_power_zero_and_recurrence(power,transpose):
    x=torch.tensor([[1.,1.,1.],[2.,2.,2.]],dtype=torch.float64)
    if transpose:x=x.T
    dec=RandomizedSVD(rank=1,power=power,random_state=7)
    u,s,vh=dec.decompose(x)
    assert torch.allclose(dec.compose(u,s,vh),x,atol=1e-13,rtol=1e-13)
    assert torch.allclose(u.mH@u,torch.eye(1,dtype=x.dtype),atol=1e-13)


@pytest.mark.parametrize('cls',[RandomizedSVD,TwoSidedRandomSVD])
def test_rsvd_size_dispatch_matches_same_random_algorithm(cls):
    for rows in [1020,1022]:
        x=torch.arange(rows*4,dtype=torch.float64).reshape(rows,4)/rows
        dec=cls(rank=2,random_state=8)
        expected=dec._decompose(x,2,random_state=15)
        actual=dec.decompose(x,2,random_state=15)
        assert torch.allclose(dec.compose(*actual),dec.compose(*expected),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('cls',[SVDDecomposition,RandomizedSVD,TwoSidedRandomSVD,CURDecomposition])
def test_zero_matrix_and_large_scale_default_rank(cls):
    for scale in [0.,1e20,1e-20]:
        x=torch.eye(3,dtype=torch.float32)*scale
        dec=cls(random_state=4)
        factors=dec.decompose(x)
        assert all(torch.isfinite(a).all() for a in factors)


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
@pytest.mark.parametrize('cls',[SVDDecomposition,RandomizedSVD,TwoSidedRandomSVD,RPHOSVDDecomposition,RSTHOSVDDecomposition,RSTDecomposition,HOOIDecomposition])
def test_complex_full_rank_reconstruction(dtype,cls):
    x=torch.tensor([[1+1j,2j],[2-1j,-1],[1j,3]],dtype=dtype)
    dec=cls(rank=2,random_state=4)
    factors=dec.decompose(x)
    assert torch.allclose(dec.compose(*factors),x,atol=3e-6 if dtype==torch.complex64 else 1e-12,rtol=3e-6 if dtype==torch.complex64 else 1e-12)


@pytest.mark.parametrize('backend',['numpy','pytorch'])
@pytest.mark.parametrize('values',[np.zeros((2,3)),np.ones((3,2)),np.diag([1.,1e-14,0.]),np.array([[1+1j,2],[2j,3],[1,-1j]])])
def test_pinv_four_moore_penrose_identities(backend,values):
    with tl.backend_context(backend):
        a=tl.tensor(values)
        pinv=pseudo_inverse(a)
        p=tl.to_numpy(pinv)
        reference=np.linalg.pinv(values,rcond=max(values.shape)*np.finfo(values.real.dtype).eps)
        assert np.allclose(p,reference,rtol=1e-10,atol=1e-10)
        assert np.allclose(values@p@values,values,atol=1e-10)
        assert np.allclose(p@values@p,p,rtol=1e-10,atol=1e-10)
        assert np.allclose((values@p).conj().T,values@p,atol=1e-10)
        assert np.allclose((p@values).conj().T,p@values,atol=1e-10)


@pytest.mark.parametrize('values,rank',[([[0.,2.],[2.,0.]],1),([[1.,1.],[1.,1.]],2),([[1.,2.],[2.,4.]],2)])
def test_cur_singular_intersection_finite_and_exact_when_rank_preserved(values,rank):
    x=torch.tensor(values,dtype=torch.float64)
    dec=CURDecomposition(rank=rank)
    c,u,r=dec.decompose(x)
    assert torch.isfinite(u).all()
    assert torch.equal(c,x[:,dec.column_indices]) and torch.equal(r,x[dec.row_indices,:])
    if rank==2:assert torch.allclose(dec.compose(c,u,r),x,atol=1e-12)


def test_cg_matches_direct_solve_preconditioners_scale_and_silence(capsys):
    base=torch.diag(torch.tensor([1.,2.,4.],dtype=torch.float64))
    for scale in [1.,1e-8,1e8]:
        a=base*scale;b=torch.ones(3,dtype=torch.float64)*scale
        for precond in [None,lambda r:r,torch.eye(3,dtype=torch.float64)]:
            x,res=conjugate_gradient(a,b,precond=precond,tol=scale*1e-13,max_iter=3)
            assert len(res)<=4
            assert torch.allclose(x,torch.linalg.solve(a,b),atol=1e-12,rtol=1e-12)
    assert capsys.readouterr().out==''


def test_cg_callable_complex_boundary_and_exhaustion():
    a=torch.tensor([[2.,1j],[-1j,3.]],dtype=torch.complex128)
    b=torch.tensor([1+1j,2-1j],dtype=torch.complex128)
    x,res=conjugate_gradient(lambda x:a@x,b,tol=1e-12,max_iter=2)
    assert torch.allclose(x,torch.linalg.solve(a,b),atol=1e-12)
    assert conjugate_gradient(a,b,x0=x,tol=1e-12)[1][0]<1e-12
    assert torch.equal(conjugate_gradient(a,torch.zeros_like(b))[0],torch.zeros_like(b))
    with pytest.raises(RuntimeError):conjugate_gradient(a,b,max_iter=0)
    _,_,info=conjugate_gradient(a,b,max_iter=0,return_info=True)
    assert not info['converged'] and info['reason']=='max_iter'
    with pytest.raises(ValueError):conjugate_gradient(-a,b)


def test_tikhonov_regularized_reference_tiny_scale_and_unsatisfied_residual():
    a=torch.tensor([[1.,2.],[3.,1.],[0.,2.]],dtype=torch.float64);b=torch.tensor([1.,2.,4.],dtype=torch.float64)
    lam=.3
    x,info=svd_solver_tikhonov(a,b,lam=lam,tol=1e-14,return_info=True)
    ref=torch.linalg.solve(a.T@a+lam**2*torch.eye(2,dtype=a.dtype),a.T@b)
    assert torch.allclose(x,ref,atol=1e-12)
    assert not info['converged'] and info['lambda']==lam and info['iterations']==1
    tiny=torch.tensor([[1e-8]],dtype=torch.float64)
    assert torch.allclose(svd_solver_tikhonov(tiny,tiny[:,0],tol=1e-16),torch.ones(1,dtype=tiny.dtype),atol=1e-12)
    with pytest.warns(RuntimeWarning):svd_solver_tikhonov(a,b,lam=lam,tol=1e-14)


@pytest.mark.parametrize('fn',[l1_norm,l2_norm,linf_norm,fro_norm,ridge_leverage])
def test_importance_real_normalized_complex_and_zero(fn):
    for x in [torch.zeros(3,2,dtype=torch.float64),torch.tensor([[1+1j,2],[2j,3],[0,-1j]],dtype=torch.complex128),torch.ones(3,2)*1e20]:
        col,row=fn(x)
        assert not col.is_complex() and not row.is_complex()
        assert (col>=0).all() and (row>=0).all()
        assert torch.isfinite(col).all() and torch.isfinite(row).all()
        assert torch.allclose(col.sum(),torch.tensor(1.,dtype=col.dtype))
        assert torch.allclose(row.sum(),torch.tensor(1.,dtype=row.dtype))


def test_ridge_leverage_float32_cancellation_regression():
    x=torch.tensor([[1.,0.],[0.,1e-4],[0.,0.]],dtype=torch.float32)
    col,row=ridge_leverage(x,lam=1e-8)
    assert torch.allclose(col,torch.tensor([2/3,1/3]),atol=1e-6)
    assert torch.allclose(row,torch.tensor([2/3,1/3,0.]),atol=1e-6)


def test_sampling_zero_weights_scale_invariance_and_boundaries():
    weights=torch.tensor([0.,1.,2.,0.],dtype=torch.float64)
    assert multinomial(weights,0,random_state=5).numel()==0
    assert set(multinomial(weights,2,random_state=3).tolist())=={1,2}
    assert torch.equal(multinomial(weights,1,random_state=4),multinomial(weights*1e-30,1,random_state=4))
    for bad in [torch.zeros(3),torch.tensor([1.,-1.]),torch.tensor([float('nan'),1.])]:
        with pytest.raises(ValueError):multinomial(bad,1,random_state=1)
    with pytest.raises(ValueError):multinomial(weights,3)
    assert topk_ids(weights,0).numel()==0 and randperm(0).numel()==0


def test_sthosvd_current_core_shapes_and_known_low_rank_roundtrip():
    rng=np.random.default_rng(3)
    factors=[np.linalg.qr(rng.normal(size=(d,2)))[0] for d in [3,4,5]]
    core=tl.tensor(rng.normal(size=(2,2,2)))
    q=[tl.tensor(a) for a in factors]
    x=ModalDecomposition.compose(core,q)
    for cls in [RPHOSVDDecomposition,RSTHOSVDDecomposition]:
        dec=cls(rank=2,random_state=8)
        result=dec.decompose(x)
        assert torch.allclose(dec.compose(*result),x,rtol=1e-12,atol=1e-12)
        for factor in result[1]:assert torch.allclose(factor.T@factor,torch.eye(2,dtype=factor.dtype),atol=1e-12)
    assert dec.diagnostics['unfolding_shapes']==[(3,20),(4,10),(5,4)]


@pytest.mark.parametrize('sampling',['uniform','norm_based','leverage_score'])
def test_rst_dependent_and_zero_factor_pinv(sampling):
    dec=RSTDecomposition(rank=3,sampling_method=sampling,random_state=3)
    for x in [torch.diag(torch.tensor([1.,1.,0.],dtype=torch.float64)),torch.zeros(3,3,dtype=torch.float64)]:
        result=dec.decompose(x)
        assert torch.isfinite(result[0]).all()
        assert torch.allclose(dec.compose(*result),x,atol=1e-12)
        assert len(dec.sample_indices)==2


@pytest.mark.parametrize('init',['svd','random'])
def test_hooi_public_arguments_initialization_and_callable_svd(init):
    x=torch.arange(60,dtype=torch.float64).reshape(3,4,5)
    dec=HOOIDecomposition(rank=[2,2,2],init=init,n_iter_max=2,random_state=4)
    core,factors=dec.decompose(x)
    assert core.shape==(2,2,2)
    dec.decompose(x,init=(core,factors),n_iter_max=1)
    dec.decompose(x,svd_type=RandomizedSVD(power=1,random_state=5).decompose,n_iter_max=1)


def test_local_decomposition_rng_interleaving_and_numpy_backend():
    x=np.arange(1.,33.).reshape(8,4)
    for backend in ['numpy','pytorch']:
        with tl.backend_context(backend):
            tensor=tl.tensor(x)
            first=RandomizedSVD(rank=2,random_state=7)
            reference=RandomizedSVD(rank=2,random_state=7)
            result=first.decompose(tensor)
            RandomizedSVD(rank=1,random_state=99).decompose(tensor)
            assert np.allclose(tl.to_numpy(first.compose(*result)),tl.to_numpy(reference.compose(*reference.decompose(tensor))),atol=1e-12)


def test_functional_explicit_keywords():
    x=torch.eye(3,dtype=torch.float64)
    for fn in [rsvd,r2svd,cur]:
        factors=fn(x,n_eigenvecs=2,random_state=3)
        assert factors[0].shape[1]==2
        with pytest.raises(TypeError):fn(x,unsupported_keyword=4)

@pytest.mark.parametrize('cls',[SVDDecomposition,RandomizedSVD,TwoSidedRandomSVD,RPHOSVDDecomposition,RSTHOSVDDecomposition,RSTDecomposition,HOOIDecomposition])
@pytest.mark.parametrize('values',[torch.zeros(0,2),torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]]),torch.ones(2,2,dtype=torch.int64),torch.ones(2,2,dtype=torch.float16),torch.ones(2,2,dtype=torch.bfloat16)])
def test_decomposition_rejects_invalid_shape_finiteness_and_dtype(cls,values):
    dec=cls(random_state=4)
    before=copy.deepcopy(dec.random_state.bit_generator.state)
    with pytest.raises((ValueError,TypeError)):dec.decompose(values)
    assert before==dec.random_state.bit_generator.state


@pytest.mark.parametrize('bad',[[0,2,2],[-1,2,2],[True,2,2],[0.0,2,2],[float('nan'),2,2],[2,2]])
def test_tensor_bad_modal_ranks_early_diagnostic(bad):
    dec=RSTHOSVDDecomposition(random_state=2)
    with pytest.raises((TypeError,ValueError)):dec.decompose(torch.ones(3,4,5),bad)


def test_local_rng_sampling_hooi_and_factorization_preserve_globals():
    np_before=np.random.get_state();torch_before=torch.random.get_rng_state().clone()
    x=torch.arange(1.,61.,dtype=torch.float64).reshape(3,4,5)
    for cls in [RPHOSVDDecomposition,RSTHOSVDDecomposition,RSTDecomposition,HOOIDecomposition]:
        dec=cls(rank=2,random_state=11,**({'init':'random','n_iter_max':2} if cls is HOOIDecomposition else {}))
        result=dec.decompose(x)
        assert torch.isfinite(result[0]).all()
    randperm(5,random_state=4)
    multinomial(torch.tensor([0.,1.,2.]),1,random_state=4)
    after=np.random.get_state()
    assert np_before[0]==after[0] and np.array_equal(np_before[1],after[1]) and np_before[2:]==after[2:]
    assert torch.equal(torch_before,torch.random.get_rng_state())

@pytest.mark.parametrize('cls',[RPHOSVDDecomposition,RSTHOSVDDecomposition,RSTDecomposition,HOOIDecomposition])
@pytest.mark.parametrize('shape,rank',[((9,2),None),((3,4,5),[3,1,1]),((3,4,5),[1,3,1])])
def test_modal_physical_cap_and_redundant_core_shape(cls,shape,rank):
    x=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
    dec=cls(rank=rank,random_state=3)
    normalized=dec._get_rank(x,None)
    result=dec.decompose(x)
    core,factors=result
    assert list(core.shape)==normalized
    assert [list(q.shape) for q in factors]==[[d,r] for d,r in zip(shape,normalized)]
    assert dec._get_rank(x,normalized)==normalized
    if rank is None:assert torch.allclose(dec.compose(*result),x,atol=1e-11,rtol=1e-11)

@pytest.mark.parametrize('backend',['numpy','pytorch'])
def test_tikhonov_tiny_lambda_large_spectrum_zero_mode(backend):
    with tl.backend_context(backend):
        a=tl.tensor(np.diag(np.array([1e20,0.],dtype=np.float32)))
        b=tl.tensor(np.ones(2,dtype=np.float32))
        x,info=svd_solver_tikhonov(a,b,lam=1e-20,return_info=True)
        assert np.isfinite(tl.to_numpy(x)).all() and np.isfinite(info['residual_norm'])
        assert np.allclose(tl.to_numpy(x),[1e-20,0.],rtol=1e-5,atol=0)
        assert not info['converged']


def test_tikhonov_rejects_nonorthogonal_fake_svd():
    with tl.backend_context('numpy'):
        a=np.array([[2.,1.],[0.,3.]])
        with pytest.raises(ValueError,match='orthonormal'):
            svd_solver_tikhonov(a,np.ones(2),svd_func=lambda _: (a,np.ones(2),np.eye(2)),lam=.1)
