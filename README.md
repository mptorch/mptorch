# MPTorch
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Overview
MPTorch is a wrapper framework built atop PyTorch that is designed to simulate the use of custom/mixed precision arithmetic in PyTorch, especially for DNN training.

It reimplements the underlying computations of commonly used layers (e.g. linear/matrix multiplication and 1D/2D/3D convolutions) so that the inputs, weights, biases and gradients of each operator can be quantized to a user-specified, per-tensor floating-point format. Quantization is opt-in per tensor: any format left unspecified simply falls back to plain PyTorch behavior.

Its main format family, `BinaryK`, is the binaryK family of IEEE P3109, the upcoming Standard for Arithmetic Formats for Machine Learning ([Fitzgibbon, Wintersteiger and Sarnoff, arXiv:2606.04028](https://arxiv.org/abs/2606.04028)): a K-bit float with P bits of precision, signed or unsigned. An explicit exponent bias also reaches formats outside the standard, such as the layouts of the OCP 8-bit E4M3 and E5M2.

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
`binaryK_quantize` simulates a binaryK format of the IEEE P3109 draft
standard: K bits in total, P of them precision (the implicit leading bit
included), e.g. K=8, P=4 is P3109's 8-bit Binary8p4, with E4M3's field widths.
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

## Documentation

The full documentation -- an explanation of the formats and rounding modes with the equations they follow, a guide per API tier, and a tutorial that trains a network in the 8-bit floating-point formats of current GPUs -- lives under `docs/` and builds with Sphinx:
```
pip3 install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```
With `uv sync` (see [Using uv](#using-uv)), that is `uv sync --extra docs` followed by `uv run sphinx-build -b html docs/source docs/_build/html`.
Every code block in it is a script under `docs/snippets/` whose recorded output is shown next to it; `python3 docs/run_snippets.py` regenerates them.

## Installation

Requirements:

- Python >= 3.12
- PyTorch >= 2.1
- GCC >= 10 on Linux (the extension is compiled with `-std=c++20`)
- CUDA >= 12.0 on Linux (only needed to build the CUDA kernels)
- Apple clang on macOS (OpenMP comes from the runtime bundled with PyTorch). The Apple GPU (MPS) kernels are built with the rest and compiled for the GPU at first use, so Xcode is not needed

MPTorch contains a C++/CUDA extension that is compiled at install time, so the build needs `torch`, `setuptools` and `ninja` already installed in the environment you install into:
```
pip install torch setuptools ninja
```

Then install MPTorch from the base directory:
```
pip install -e . --no-build-isolation
```

Always pass `--no-build-isolation`. Without it, pip builds the extension in a temporary environment with its own freshly resolved copy of `torch`, which may not be the build installed in your environment. The compiled extension then fails at import with an `undefined symbol` error, because it was built against a different `torch` ABI than the one loading it.

By default, the CUDA extension is built whenever `torch.cuda.is_available()` and `CUDA_HOME` are set, and on macOS the MPS kernels whenever PyTorch was built with MPS (`USE_MPS=0` leaves them out). On an `"mps"` tensor every op returns the CPU's result bit for bit, stochastic rounding included, up to the GPU's flush of binary32 subnormals (see the documentation's "On an Apple GPU"). To force a CPU-only build:
```
USE_CUDA=0 pip install -e . --no-build-isolation
```

To build with debug symbols and no optimization:
```
DEBUG=1 pip install -e . --no-build-isolation
```

The compiled objects are kept under `build/`, so running the install command again after a change recompiles only the files the change reaches, and takes seconds when nothing changed. Delete `build/` to start from scratch. If [ccache](https://ccache.dev) is installed, PyTorch's extension builder uses it without further setup, and a build from scratch then reuses whatever it has compiled before.

### Using uv

[uv](https://docs.astral.sh/uv/) can install MPTorch in two ways: as a project, with `uv sync`, or in place of pip, with `uv pip`.

#### With `uv sync`

`pyproject.toml` configures uv's project commands, so from the base directory
```
uv sync --extra test
```
creates `.venv` if there is none, installs the dependencies, and then builds MPTorch against the `torch` it has just installed and installs it in editable mode. The project's settings already build without isolation, so there is no `--no-build-isolation` to pass. `torch` and `torchvision` come from PyTorch's CUDA 13.2 wheel index on Linux and from PyPI on macOS, and everything else from PyPI. The versions are pinned in `uv.lock`, which `uv sync` writes the first time and follows afterwards. The extras are `test` (the test suite), `docs` (the documentation) and `dev` (the linters); select the ones you need. Commands then run in that environment through `uv run`:
```
uv run pytest tests/
uv run python my_script.py
```

`uv sync` and `uv run` rebuild the extension when a file under `mptorch/csrc/`, `setup.py` or a requirements file changes, and when one of the build variables changes (`USE_CUDA`, `USE_MPS`, `DEBUG`, `MPTORCH_NO_FP64`, `MPTORCH_COMPRESS_FATBIN`). Because the variables are part of that check, export one for the whole session rather than setting it on a single command: after `DEBUG=1 uv sync`, a plain `uv run` finds `DEBUG` unset and rebuilds without it.

`uv sync` makes the environment match the lock exactly, uninstalling any package the selected extras do not list; `uv sync --inexact` leaves such packages installed. `uv run` never uninstalls anything. To move to the newest `torch`, run `uv lock --upgrade-package torch --upgrade-package torchvision` and then `uv sync`, which recompiles the whole extension against it.

The CUDA toolkit that compiles the extension has to match the CUDA version of the `torch` build. To use a toolkit other than 13.x, point `[[tool.uv.index]]` in `pyproject.toml` at the matching PyTorch index (for instance `https://download.pytorch.org/whl/cu129` for CUDA 12.9).

#### With `uv pip`

Every `pip install` command in this README works the same with `uv pip install` in its place. A virtual environment created by `uv venv` doesn't contain `pip` at all, so there `uv pip` is the one to use:
```
uv venv
uv pip install torch setuptools ninja
uv pip install -e . --no-build-isolation
```
`uv pip` installs into the activated environment, or into `.venv` in the current directory if none is activated. `--no-build-isolation` is just as necessary with `uv pip` as with `pip`. Run from the base directory, `uv pip` also looks for `torch` on PyTorch's CUDA 13.2 index (the `[tool.uv.pip]` table in `pyproject.toml`). Unlike `uv sync`, `uv pip install -e .` rebuilds every time it runs; since the compiled objects are kept under `build/`, that takes seconds when nothing changed.

### Running the tests

The test suite has a couple of extra dependencies (`pytest`, `gfloat`) that aren't required to just use the library, so they're kept in a separate `test` extra. Install it with the same `--no-build-isolation` flag (with `uv pip install` in place of `pip install` if you use `uv pip`), then run the suite:
```
pip install -e ".[test]" --no-build-isolation
pytest tests/
```
With `uv sync`, the same is `uv sync --extra test` followed by `uv run pytest tests/`.
On a machine without a CUDA device, the CUDA tests are skipped automatically, and so are the MPS tests without an Apple GPU.

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
