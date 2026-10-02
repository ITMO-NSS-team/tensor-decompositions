import math
import numpy
import torch
import tensorly as tl

from tdecomp.matrix.random_projections import sparse_jl_matrix
tl.set_backend('pytorch')

w = torch.arange(24).reshape(2, 4, 3)
wt = w.T
print(w, w.T)

w = torch.arange(24).reshape(3, 8)
conditioner = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1])

mul_res = torch.mul(w, conditioner)
mul_res2 = torch.matmul(w, conditioner)
print(mul_res)

C = torch.rand((3, 3))
print("C", C)
c_inv_torch = torch.linalg.pinv(C)
print("torch", c_inv_torch)
c_inv_solve = tl.solve(tl.matmul(C.T, C), C.T)
print("ours", c_inv_solve)
print(C @ c_inv_torch)
print(C @ c_inv_solve)

print(min(tl.shape(C)))


def estimate_stable_rank(W):
        n_samples = max(tl.shape(W))
        eps = 0.5
        min_num_samples = torch.ceil(4 * torch.log(torch.scalar_tensor(n_samples)) / (eps**2 / 2 - eps**3 / 3))
        return max(min(torch.round(min_num_samples), *tl.shape(W)), 1)
                   
print("st rank", estimate_stable_rank(torch.eye(8)))
# U, S, Vh = tl.svd_interface(C)
# print("u", U.T)
# print('vh', Vh.T)
# print(S)
# print("s inv", tl.diag(1 / S))
# print("s * u", tl.matmul(tl.diag(1 / S), U.T))
# print("v * s * u", tl.matmul(Vh.T, tl.matmul(tl.diag(1 / S), U.T)))
# c_inv_SVD = tl.matmul(Vh.T, tl.matmul(tl.diag(1 / S), U.T))
# print(c_inv_SVD, c_inv_torch)
# print(torch.dist(c_inv_torch, c_inv_solve))
# print(tl.solve(tl.matmul(C, C.T), C))

print(tl.matmul(torch.rand(3, 3, 2), tl.reshape(torch.tensor([[1, 0], [0, 0], [0, 0]]) * 1.0, (1, 2, 3))))
S = torch.tensor([1, 0, 1, 1])
matr = torch.rand(3, 4, 4)
# print(tl.matmul(matr, tl.reshape(S, (1, 4))))
a = numpy.ones((5, 5))
print(a[None, ...])

print(tl.matmul(torch.rand(3, 5, 8, 7), tl.reshape(torch.rand(5, 8, 7), (1, 5, 7, 8))).shape)
print(tl.reshape(matr, shape=[1] + list(tl.shape(matr)),))
print(type(tl.shape(matr)))

factors = [torch.rand(3, 3) for i in range(3)]
core = torch.rand(3, 3, 3)
#check that mode_dot works
for i, factor in enumerate(factors):
    core = tl.tenalg.mode_dot(core, factor, i)
print(core)

tl.SVD_FUNS
C2 = torch.rand(1000, 100)
u, s, vh = tl.svd_interface(C2, n_eigenvecs=min(tl.shape(C2)), full_matrices=False)
print("torch back", u.shape, s.shape, vh.shape)
print("U", u)
C1 = C2.numpy()
tl.set_backend('numpy')
u, s, vh = tl.svd_interface(C1, n_eigenvecs=min(tl.shape(C1)), flip_sign=False, full_matrices=False)
print("numpy U", u, "numpy v", vh)
print("numpy back", u.shape, s.shape, vh.shape)
tl.set_backend('pytorch')
u, s, vh = torch.linalg.svd(C2, full_matrices=True)
print("full true", u.shape, s.shape, vh.shape)
print("u full", u)
u, s, vh = torch.linalg.svd(C2, full_matrices=False)
print("full false", u.shape, s.shape, vh.shape)
print("U non full", u)
print("vh not full", vh)
C3 = torch.tensor([[1, 0], [1, 0]]) * 1.0
u, s, vh = tl.svd_interface(C3, n_eigenvecs=min(tl.shape(C3)), flip_sign=False, full_matrices=False)
print(u, s, vh)
print(torch.linalg.svd(C3, full_matrices=False))
print(tl.svd_interface(C3, n_eigenvecs=min(tl.shape(C3)), flip_sign=False, full_matrices=False)[1])
print(C2.size(-2), C2.size())
# A = torch.randn(7, 5, 3)
# U, S, Vh = torch.linalg.svd(A, full_matrices=False)
# print(torch.dist(A, U @ torch.diag_embed(S) @ Vh), U.shape, S.shape, Vh.shape) #просто батчевый svd, не настоящий

tl.set_backend("pytorch")
a = tl.random.random_tensor((3, 3))
for i in range(100):
    a += tl.random.random_tensor((3, 3))
print(a / 101, a)
context = {"dtype": torch.int}
print(tl.tensor(a, **context))
print(tl.int32(a))

tl.set_backend("numpy")
a = tl.random.random_tensor((3, 3)) * 2 * 3
R = tl.int32(a)
print(type(a), type(R), "WARN: tl.int32 cast to numpy ndarray int32! not torch!")
# R_ver2 = tl.tensor(tl.random.random_tensor((3, 3)) * 2 * 3, dtype=tl.int32)
# print("r ver 2", R_ver2, type(R_ver2)) NOTE doesnt work!
# b = tl.tensor(a, dtype=tl.int32)
# print("to", b, type(b))
# print("R", R)
mul = tl.int32(R == 0) * math.sqrt(3)
print(mul, type(R), type(mul), mul.dtype) #sparse_iid_entries checked
print(a // 1, ((a // 1 == 0) * 1), type(((a // 1 == 0) * 1))) #решение!
tl.set_backend("pytorch")

a = torch.empty(5, 2)
print(tl.reshape(a, (-1,)))
print(tl.arange(5))
print(tl.zeros((5 * 3)))
print(torch.repeat_interleave(torch.tensor([2, 5, 10]), 2))
k = 5
s = 3
cols = tl.tensor([0] * (k * s)) #torch tensor создаётся!
for i in range(0, k):
    for j in range(s):
        tl.index_update(cols, tl.index[s * i + j], i)
print("cols", cols)
print(tl.zeros((5, 2)))
a = torch.arange(8)
tl.index_update(a, tl.index[4], torch.tensor(1000))
print(a)


d = 10
rows = tl.tensor((tl.random.random_tensor((k, s)) * d // 1).reshape(-1), dtype=cols.dtype)
R = tl.zeros((d, k))

values = tl.random.random_tensor((k, s), **context) * 4 // 1 - 1.0
    
values *= math.sqrt(1 / s)
print("values", values)
print("cols", cols)
print("rows", rows)

tl.index_update(R, tl.index[rows, cols], tl.reshape(values, (-1,)))
# print(R) #DONE!

tl.set_backend("numpy")
#WARN! context should be generated once, set_backend only used in initial stage when link library!
#after that some consts will not be changed, for example tl.float32
print("check sparse numpy", sparse_jl_matrix(5, 3, context={}))
tl.set_backend("pytorch")
print("check sparse torch", sparse_jl_matrix(5, 3))