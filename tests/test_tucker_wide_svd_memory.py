import torch
from experiments.hypotheses import synthetic_tucker_common as common


def test_wide_spatial_unfolding_never_builds_square_right_basis(monkeypatch):
    native=torch.linalg.svd;calls=[]
    def tracked(matrix,**kwargs):
        calls.append((matrix.shape,kwargs['full_matrices']))
        return native(matrix,**kwargs)
    monkeypatch.setattr(torch.linalg,'svd',tracked)
    weight=torch.randn(32,32,3,3,dtype=torch.float64)
    core,factors=common.exact_hosvd(weight,(8,8,3,3))
    assert all(not full for shape,full in calls if shape[1]>shape[0])
    assert any(shape==(3,3072) for shape,full in calls)
    assert core.shape==(8,8,3,3)
    for factor in factors:
        torch.testing.assert_close(factor.T@factor,torch.eye(factor.shape[1],dtype=torch.float64))


def test_tall_rank_deficient_full_modal_basis_is_preserved():
    weight=torch.arange(8,dtype=torch.float64).reshape(8,1,1,1)
    core,factors=common.exact_hosvd(weight,(8,1,1,1))
    assert factors[0].shape==(8,8)
    torch.testing.assert_close(common.reconstruct(core,factors),weight)


def test_tall_low_rank_does_not_construct_unused_complete_left_basis(monkeypatch):
    native=torch.linalg.svd;calls=[]
    def tracked(matrix,**kwargs):
        calls.append((matrix.shape,kwargs['full_matrices']))
        return native(matrix,**kwargs)
    monkeypatch.setattr(torch.linalg,'svd',tracked)
    common.exact_hosvd(torch.arange(100,dtype=torch.float64).reshape(100,1,1,1),(1,1,1,1))
    assert calls[0]==(torch.Size([100,1]),False)
