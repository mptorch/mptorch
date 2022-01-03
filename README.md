# MPTorch
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Overview
MPTorch is a wrapper framework built atop PyTorch that is
designed to simulate the use of custom/mixed precision
arithmetic in PyTorch, especially for DNN training.

It reimplements many of the underlying computations of 
commonly used layers for CNNs (e.g., matrix multiplication
and batch normalization) using user-specified floating-point formats for each operation (e.g., addition, multiplication). All the operations are internally done using IEEE-754 32-bit floating point
arithmetic, with the results rounded to the specified format.

MPTorch is still in its early stages of development, but it
is already capable of training convolutional neural networks using custom floating-point formats that are specified at the layer level (and for every operator type) for both forward and backward pass computations.

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