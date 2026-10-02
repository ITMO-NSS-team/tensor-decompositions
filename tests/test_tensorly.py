import numpy
import pytest
import tensorly as tl
import torch

import tdecomp

def test_mode_dot():
    tl.set_backend("pytorch")
    factors = [torch.rand(3, 3) for i in range(3)]
    core = torch.rand(3, 3, 3)
    init_shape = tl.shape(core)
    #check that mode_dot works
    for i, factor in enumerate(factors):
        core = tl.tenalg.mode_dot(core, factor, i)
    assert tl.shape(core) == init_shape

def test_batched_matmul_and_reshape():
    tl.set_backend("pytorch")
    res1 = tl.matmul(torch.rand(3, 5, 8, 7), tl.reshape(torch.rand(5, 8, 7), (1, 5, 7, 8)))
    assert tl.shape(res1) == (3, 5, 8, 8)
    with pytest.raises(RuntimeError): #batched shapes doesnt match
        tl.matmul(torch.rand(3, 5, 8, 7), torch.rand(1, 6, 7, 8))
    
    with pytest.raises(RuntimeError): #batched shapes doesnt match
        tl.matmul(torch.rand(3, 5, 8, 7), torch.rand(4, 5, 7, 8))


def test_incorrect_tl_int32():
    tl.set_backend("pytorch")
    a = tl.random.random_tensor((3, 3))
    a = tl.int32(a)
    assert type(a) is not torch.Tensor
    assert type(a) is numpy.ndarray

    with pytest.raises(ValueError):
        tl.tensor(tl.random.random_tensor((3, 3)), dtype=tl.int32)

    tl.set_backend("numpy")
    a = tl.random.random_tensor((3, 3))
    a = tl.int32(a)
    assert type(a) is numpy.ndarray

def test_tl_float32():
    tl.set_backend("pytorch")
    assert tl.float32 is torch.float32

def test_type_casts():
    tl.set_backend("pytorch")
    a = tl.random.random_tensor((3, 3))
    assert a.dtype is torch.float64
    a = a * 5
    assert a.dtype is torch.float64
    a = a // 1
    assert a.dtype is torch.float64
    a = (a == 0)
    assert a.dtype is torch.bool
    a = a * 1
    assert a.dtype is torch.int64
    b = tl.tensor([0] * (3))
    assert b.dtype is torch.int64
    a = tl.random.random_tensor((3, 3))
    assert tl.tensor(a, **tl.context(b)).dtype == b.dtype


def test_reshape():
    tl.set_backend("pytorch")
    a = torch.empty(5, 2)
    assert tl.shape(tl.reshape(a, (-1,))) == (10,)
    with pytest.raises(TypeError):
        tl.reshape(a, (-1))

    with pytest.raises(TypeError):
        tl.reshape(a, (-1,), copy=False)

def test_argsort():
    tl.set_backend("pytorch")
    arr = torch.arange(30, 20, -1)

    with pytest.raises(RuntimeError):
        ind = tl.argsort(arr)

    with pytest.raises(TypeError):
        ind = tl.argsort(arr, dim=-1)

    ind = tl.argsort(arr, -1)
    assert ind[0] == 9
    tl.set_backend("numpy")
    ind = tl.argsort(arr, -1)
    assert ind[0] == 9

def test_sort():
    tl.set_backend("pytorch")
    arr = tl.arange(30, 20, -1)
    assert tl.sort(arr, -1)[0] == 21
    
    with pytest.raises(TypeError):
        ind = tl.sort(arr)

    assert (tl.sort(arr, axis=0) == tl.sort(arr, -1)).all()

    tl.set_backend("numpy")
    arr = tl.arange(30, 20, -1)
    assert tl.sort(arr, -1)[0] == 21