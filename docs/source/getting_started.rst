Getting started
===============

Requirements
------------

- Python >= 3.12 and PyTorch >= 2.1
- A C++ compiler (GCC >= 4.9 on Linux)
- CUDA >= 12.0, only if the CUDA kernels are wanted
- On macOS, nothing more: the Apple GPU (MPS) kernels are built with the rest, and compiled for the GPU when first used, so Xcode is not needed

MPTorch is a C++/CUDA extension, so installing it compiles the kernels against the PyTorch you have installed.

Installation
------------

From the root of the repository:

.. code-block:: console

   $ pip install -e . --no-build-isolation

or 

.. code-block:: console

   $ uv pip install -e . --no-build-isolation

if you have ``uv`` on your machine.

Always pass ``--no-build-isolation``. Without it, ``pip`` builds the extension in a throw-away environment against a *freshly downloaded* PyTorch that may not be the one you run, and the result fails to import with an ``undefined symbol`` error. With it, the build uses your installed PyTorch.

The CUDA kernels are built whenever ``torch.cuda.is_available()`` is true and ``CUDA_HOME`` is set, and the Apple GPU kernels on macOS whenever PyTorch was built with MPS (``USE_MPS=0`` leaves them out). To force a CPU-only build, or a debug build:

.. code-block:: console

   $ USE_CUDA=0 pip install -e . --no-build-isolation
   $ DEBUG=1 pip install -e . --no-build-isolation

The test suite needs two extra packages, kept in a ``test`` extra:

.. code-block:: console

   $ pip install -e ".[test]" --no-build-isolation
   $ pytest tests/

Again, these commands can also be executed using their ``uv`` equivalents:

.. code-block:: console

   $ USE_CUDA=0 uv pip install -e . --no-build-isolation
   $ DEBUG=1 uv pip install -e . --no-build-isolation

and

.. code-block:: console

   $ uv pip install -e ".[test]" --no-build-isolation
   $ pytest tests/

A first example
---------------

Two objects carry most of the library: a *format*, which is a plain value, and a *quantizer*, which is that format turned into a function on tensors. Layers take one quantizer per signal, and anything left unset is plain PyTorch.

.. literalinclude:: ../snippets/getting_started.py
   :language: python
   :caption: docs/snippets/getting_started.py

.. literalinclude:: ../snippets/getting_started.out
   :language: text
   :caption: output

``BinaryK(8, 4)`` is ``Binary8p4se`` from IEEE P3109, the upcoming standard for machine-learning number formats: an 8-bit float with a sign bit, 4 exponent bits and 4 bits of precision (one of them implicit). Those are the field widths of the OCP format E4M3, whose exponent bias is one less. ``1.541`` becomes ``1.5`` because the values representable between 1 and 2 are spaced ``1/8`` apart. :doc:`concepts` explains the formats and the rounding; the guides take it from there.

Vocabulary
----------

A few words are used throughout, with a fixed meaning:

format
   A description of which numbers can be represented: :class:`~mptorch.BinaryK` (the IEEE P3109 binary formats) or :class:`~mptorch.SuperFP`, for example. Frozen, hashable, comparable by value.
rounding mode
   How a number that is *not* representable is mapped to one that is: :class:`~mptorch.RoundMode`. A property of an operation, not of a format.
quantize / quantizer
   To round every element of a tensor to a format; the function that does it. In MPTorch this is always "compute in the carrier, round the result".
carrier
   The IEEE 754 format a simulation computes and rounds in: by default binary32 for float32, float16 and bfloat16 tensors, and binary64 for float64 ones. It bounds which formats can be simulated whole. A carrier, named ``torch.float32`` or ``torch.float64``, is never narrower than the dtype of the tensor it computes, and the result keeps the tensor's dtype (see :doc:`concepts`).
signal
   One of the tensors flowing through a layer: its input, weight, bias, the gradient arriving from above, and the gradients it produces.
arithmetic
   What happens *inside* a matrix product: which format each partial product and each running sum is rounded to. This is the part a plain quantizer cannot express, and the part MPTorch's kernels exist for.
MAC
   Multiply-accumulate. The arithmetic of one step of a dot product, as a value: :class:`~mptorch.quant.SplitMac` (the product and the sum rounded separately) or :class:`~mptorch.quant.FusedMac` (one fused multiply-add rounded once), with an *accumulate algorithm* saying in what order the terms are summed (see :doc:`kernels/arithmetic`).
block format
   Narrow element codes that share one scale per *block* of consecutive elements, or per *tile* of rows of blocks: :class:`~mptorch.BlockFormat`, which covers OCP's MX formats and NVIDIA's NVFP4 (see :ref:`block-formats`). A tensor in one is *packed* in the format's own bytes.
pass
   One of the matrix products of a training step: the forward, and the two gradient products of the backward pass. Each runs in the arithmetic of its own hook, and each is a product of the one core routine (see :doc:`kernels/index`).
