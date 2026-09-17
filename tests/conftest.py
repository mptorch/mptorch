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
