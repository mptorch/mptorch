import pytest
import torch


def parametrize_seed(*seeds):
    """Run the test once per seed, each through conftest.py's autouse ``seed``
    fixture: the parametrization is indirect, so the fixture seeds torch with
    the value before the test body runs. Without it every test runs at 1234."""
    return pytest.mark.parametrize("seed", seeds, indirect=True)


def _mps_works() -> bool:
    """Whether there is an Apple GPU that runs. ``is_available()`` alone is not
    enough: GitHub's macOS runners report MPS as available and then fail every
    allocation on it, which would fail the MPS tests there instead of skipping
    them."""
    if not torch.backends.mps.is_available():
        return False
    try:
        return torch.ones(2, device="mps").sum().item() == 2.0
    except RuntimeError:
        return False


HAS_MPS = _mps_works()

# Skips a test that needs a GPU when none is visible. Use these and the device
# lists below rather than a hand-rolled `torch.cuda.is_available()` check, so a
# machine without the device skips every such case with the same reason.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA-capable device found."
)

# The same for an Apple GPU (the MPS backend, csrc/mps/).
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="No working MPS device found.")

_cuda = pytest.param("cuda", marks=requires_cuda)
_mps = pytest.param("mps", marks=requires_mps)

# Values for a `device` parameter: "cpu" always, and "cuda" and "mps" skipped
# where there is no such device. `@pytest.mark.parametrize("device",
# available_devices)` is the convention for a test that runs on every backend.
available_devices = ["cpu", _cuda, _mps]

# The same for a test that needs float64 tensors on the device, which MPS does
# not have: the binary64 carrier, and any float64 reference computed where the
# operands are. A test parametrized over both a device and a dtype takes
# `available_devices` instead, and conftest.py skips its (mps, float64) cases.
float64_devices = ["cpu", _cuda]

# The same for a test that only makes sense on CUDA but keeps a `device`
# parameter, so its ids and signature match the multi-backend tests.
cuda_devices = [_cuda]


def has_float64(device: str) -> bool:
    """Whether `device` has float64 tensors, for a test that loops over both
    carriers inside one case and must leave binary64's out on MPS."""
    return device != "mps"
