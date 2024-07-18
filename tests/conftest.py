import torch
import pytest

@pytest.fixture(scope="function", autouse=True)
def seed(request):
    if hasattr(request, "param"):
        value = request.param
    else:
        value = 1234  # default seed
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    return value