import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def seed(request):
    """Seed torch's CPU and CUDA generators and make cuDNN deterministic before
    every test, so random inputs and stochastic-rounding draws repeat from run
    to run. The seed is 1234 unless the test carries
    ``tests.markers.parametrize_seed(...)``, which supplies each of its values
    through ``request.param``. Returns the seed in use, for a test that wants
    to seed a generator of its own the same way."""
    if hasattr(request, "param"):
        value = request.param
    else:
        value = 1234  # no parametrize_seed on the test
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    return value


@pytest.fixture(autouse=True)
def _no_float64_on_mps(request):
    """Skip the MPS cases of a test parametrized over a device and a float64
    dtype or the binary64 carrier (a parameter that is ``torch.float64`` or
    the string ``"binary64"``): MPS has no float64 tensors, so there is
    nothing to run. A test that needs float64 on its device whatever its
    parameters takes ``tests.markers.float64_devices`` instead."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if params.get("device") == "mps" and any(
        v is torch.float64 or v == "binary64" for v in params.values()
    ):
        pytest.skip("MPS has no float64 tensors.")
