"""Keep legacy tests explicit about backend and isolate their configuration."""
import pytest
import tensorly as tl


@pytest.fixture(autouse=True)
def tensorly_backend():
    previous = tl.get_backend()
    tl.set_backend("pytorch")
    try:
        yield
    finally:
        tl.set_backend(previous)
