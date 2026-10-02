"""Composition containers for linear and modal factorizations."""
import abc
from functools import reduce
import tensorly as tl

__all__ = ['LinearDecomposition', 'ModalDecomposition']


class _IDecompositionResult(abc.ABC):
    def __init__(self, tensors):
        if isinstance(tensors, dict):
            self.tensors = dict(tensors)
        elif isinstance(tensors, (tuple, list)):
            self.tensors = dict(enumerate(tensors))
        else:
            raise TypeError('tensors must be a dictionary or sequence')

    @classmethod
    @abc.abstractmethod
    def compose(cls, *tensors):
        pass


class LinearDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, *tensors):
        if not tensors:
            raise ValueError('At least one factor is required')
        if len(tensors) == 3 and tl.ndim(tensors[1]) == 1:
            u, s, vh = tensors
            return tl.matmul(u * s, vh)
        return reduce(tl.matmul, tensors)


class ModalDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, core, *factors):
        """Compose Tucker core with a factor list or unpacked mode factors."""
        if len(factors) == 1 and isinstance(factors[0], (list, tuple)):
            factors = factors[0]
        if len(factors) != tl.ndim(core):
            raise ValueError('One factor is required per core mode')
        for mode, factor in enumerate(factors):
            if tl.ndim(factor) != 2 or tl.shape(factor)[1] != tl.shape(core)[mode]:
                raise ValueError('Each factor must have shape (mode_size, core_mode_size)')
        return tl.tenalg.multi_mode_dot(core, factors)
