import pytest
import torch


def parametrize_seed(*seeds):
    """Run the test once per seed, each through conftest.py's autouse ``seed``
    fixture: the parametrization is indirect, so the fixture seeds torch with
    the value before the test body runs. Without it every test runs at 1234."""
    return pytest.mark.parametrize("seed", seeds, indirect=True)


# Skips a test that needs a GPU when none is visible. Use this and the device
# lists below rather than a hand-rolled `torch.cuda.is_available()` check, so a
# CPU-only machine skips every CUDA case with the same reason.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA-capable device found."
)

# Values for a `device` parameter: "cpu" always, and "cuda" skipped when no
# device is available. `@pytest.mark.parametrize("device", available_devices)`
# is the convention for a test that runs on both backends.
available_devices = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="No CUDA-capable device found."
        ),
    ),
]

# The same for a test that only makes sense on CUDA but keeps a `device`
# parameter, so its ids and signature match the two-backend tests.
cuda_devices = [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="No CUDA-capable device found."
        ),
    )
]
