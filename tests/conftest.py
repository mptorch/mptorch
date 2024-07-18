import torch
import pytest

@pytest.fixture(scope="function", autouse=True)
def determinism():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True