from typing import Any, Callable, Literal, TypeAlias, Union

TensorLike: TypeAlias = Any
'''Tensorly supports work with different tensor backends (numpy, torch.tensor and so on),
but it doesnt describe abstract class for it.
So the tensor can be of `Any` type depending on backend setted in `tl.set_backend` .'''

Number = Union[int, float]
'''Type, widely used for 'rank' typing'''

TensorDecompositionInit: TypeAlias = tuple[TensorLike, list[TensorLike]] | Literal['svd', 'random']
'''Used in iterative tensor decomposition algorithms to determine (start factorization)/(algorithm for start factorization) that will be optimized'''

SVDCallable: TypeAlias = Callable[[TensorLike], tuple[TensorLike, TensorLike, TensorLike]]

def __getattr__(name):
    """Resolve legacy backend-specific dtype names at access time."""
    if name in {"BOOL_TYPE", "COMPLEX64_TYPE"}:
        import tensorly as tl
        return tl.tensor([True]).dtype if name == "BOOL_TYPE" else tl.backend.complex64
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
