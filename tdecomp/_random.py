"""Local random streams shared by sampling and sketch generators.

Integers create a stream; NumPy Generator/RandomState is consumed in place.
None retains legacy global NumPy randomness. No torch RNG is consumed.
"""
import numbers
import numpy as np


def normalize_random_state(random_state=None):
    if random_state is None:
        return np.random
    if random_state is np.random:
        return random_state
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    if isinstance(random_state, numbers.Integral) and not isinstance(random_state, bool):
        return np.random.default_rng(int(random_state))
    raise TypeError("random_state must be None, integer, Generator, or RandomState")


def rng_integers(rng, low, high=None, size=None):
    if hasattr(rng, 'integers'):
        return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size)
