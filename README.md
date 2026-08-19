# MPTorch
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Overview
MPTorch is a wrapper framework built atop PyTorch that is designed to simulate the use of custom/mixed precision arithmetic in PyTorch, especially for DNN training.

It reimplements the underlying computations of commonly used layers (e.g. linear/matrix multiplication and 1D/2D/3D convolutions) so that the inputs, weights, biases and gradients of each operator can be quantized to a user-specified, per-tensor floating-point format. Quantization is opt-in per tensor: any format left unspecified simply falls back to plain PyTorch behavior.

MPTorch is still in its early stages of development, but it is already capable of training neural networks using custom floating-point formats that are specified at the layer level (and for every operator's inputs, outputs and gradients) for both forward and backward pass computations.

## Basic usage example
The code is supposed to be straightforward to write for users familiar with PyTorch. The following example illustrates how a simple MLP can be built and trained on MNIST with a custom, narrow floating-point format applied to its weights, activations and gradients:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mptorch.number import RoundMode
import mptorch.quant as qpt

"""Hyperparameters"""
batch_size = 64  # batch size
lr_init = 0.05  # initial learning rate
num_epochs = 10  # epochs
momentum = 0.9
weight_decay = 0

"""Prepare the transforms on the dataset"""
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

"""download dataset: MNIST"""
train_dataset = datasets.MNIST(
    "./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    "./data", train=False, transform=transform, download=False
)
test_loader = DataLoader(
    test_dataset, batch_size=int(batch_size), shuffle=False
)

"""
Specify the quantization function shared by every signal
(weights, activations, biases and gradients) in the layers below.
`binaryK_quantize` simulates a K-bit floating-point format with
P mantissa (precision) bits, e.g. K=8, P=4 is a narrow 8-bit float.
"""
K, P = 8, 4
quant_fp = lambda x: qpt.binaryK_quantize(x, K=K, P=P, rounding_mode=RoundMode.RNE)

layer_formats = qpt.QAffineFormats(
    weight_quant=quant_fp,
    input_quant=quant_fp,
    bias_quant=quant_fp,
    wgrad_quant=quant_fp,
    igrad_quant=quant_fp,
    bgrad_quant=quant_fp,
)

"""Construct the model"""
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)


model = nn.Sequential(
    Reshape(),
    qpt.QLinear(784, 128, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(128, 96, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(96, 10, formats=layer_formats),
)

"""Prepare and launch the training process"""
model = model.to(device)
optimizer = SGD(
    model.parameters(),
    lr=lr_init,
    momentum=momentum,
    weight_decay=weight_decay,
)

for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(dim=1) == target).sum().item()
    print(f"epoch {epoch}: test accuracy = {correct / len(test_dataset):.4f}")

```

`QAffineFormats` also accepts `fwd_math`, `bwd_igrad_math` and `bwd_wgrad_math` callables for cases where the forward/backward arithmetic itself (not just the tensors going into it) needs to be simulated with a custom, non-default routine; leaving them unset (as above) falls back to plain PyTorch's `F.linear`/`F.conv*d` and autograd.

## Installation

Requirements:

- Python >= 3.12
- PyTorch >= 2.1
- GCC >= 4.9 on Linux
- CUDA >= 12.0 on Linux (only needed to build the CUDA kernels)

Install MPTorch through pip (from the base directory):
```
pip3 install -e .
```

By default, the CUDA extension is built whenever `torch.cuda.is_available()` and `CUDA_HOME` are set. To force a CPU-only build:
```
USE_CUDA=0 pip3 install -e .
```

To build with debug symbols and no optimization:
```
DEBUG=1 pip3 install -e .
```

### Running the tests

The test suite has a couple of extra dependencies (`pytest`, `gfloat`) that aren't required to just use the library, so they're kept in a separate `test` extra:
```
pip3 install -e ".[test]"
pytest tests/
```

## Acknowledgements
This project is based on the same logic that is used
in [QPyTorch](https://github.com/Tiiiger/QPyTorch) and [CPD](https://github.com/drcut/CPD).

## Team
- [@sfilip](https://github.com/sfilip)
- [@ubc-guy](https://github.com/ubc-guy)
- [@samiBENALI](https://github.com/samiBENALI)
- [@bob-0727](https://github.com/bob-0727)
- [@carasol0217](https://github.com/carasol0217)
- [@shingwaipun](https://github.com/shingwaipun)
- [@Krafpy](https://github.com/Krafpy)
- [@BobbbbbZ](https://github.com/BobbbbbZ)
- [@nirvik-pande](https://github.com/nirvik-pande)
- [@VictorRavain](https://github.com/VictorRavain)
