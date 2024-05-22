# MPTorch
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Overview
MPTorch is a wrapper framework built atop PyTorch that is designed to simulate the use of custom/mixed precision arithmetic in PyTorch, especially for DNN training.

It reimplements the underlying computations of commonly used layers for CNNs (e.g. matrix multiplication and 2D convolutions) using user-specified floating-point 
formats for each operation (e.g. addition, multiplication). All the operations are internally done using IEEE-754 32-bit floating-point arithmetic, with the results rounded to the specified format.

MPTorch is still in its early stages of development, but it is already capable of training convolutional neural networks using custom floating-point formats that are specified at the layer level (and for every operator type) for both forward and backward pass computations.

## Basic usage example
The code is supposed to be straightforward to write for users familiar with PyTorch. The following example illustrates how a simple MLP example can be run:

```python
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
from mptorch.utils import trainer

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
Specify the formats and quantization functions 
for the layer operations and signals
"""
exp, man = 4, 2
fp_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
quant_fp = lambda x: qpt.float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False)
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(fp_format, fp_format),
    fwd_rnd="nearest",
    bwd_mac=(fp_format, fp_format),
    bwd_rnd="nearest",
    weight_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
    bias_quant=quant_fp,
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

"""
Specify the format to be used for updating model parameters
"""
acc_q = lambda x: qpt.float_quantize(
    x, exp=8, man=15, rounding="nearest"
)
optimizer = OptimMP(
    optimizer,
    weight_quant=weight_q,
    acc_quant=acc_q,
    momentum_quant=acc_q,
)

"""
Utility function used to train the model (loss scaling is
supported)
"""
trainer(
    model,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=lr_init,
    batch_size=batch_size,
    optimizer=optimizer,
    device=device,
    init_scale=1024.0,
)

```

## Installation

Requirements:

- Python >= 3.6
- PyTorch >= 1.5.0
- GCC >= 4.9 on Linux
- CUDA >= 10.1 on Linux

Install other requirements by:
```bash
pip install -r requirements.txt
```

Install MPTorch through pip (from the base directory):
```
pip install -e .
```

## Acknowledgements
This project is based on the same logic that is used
in [QPyTorch](https://github.com/Tiiiger/QPyTorch) and [CPD](https://github.com/drcut/CPD).

## Team
- [Silviu Filip](https://people.irisa.fr/Silviu-Ioan.Filip/)
- [Wassim Seifeddine](https://wassimseifeddine.com/)