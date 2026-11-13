import os

import pytest
import torch

# Triton 3.9 contracts a multiply and an add into an FMA in the kernels Inductor
# generates around the ops (our kernels are not Triton's), unless told
# otherwise, and warns that 3.10 stops by default. The suite takes 3.10's
# default now, which is also eager's: eager runs the two as separate kernels.
# Triton reads it when it compiles a kernel, after this module is imported.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")


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
