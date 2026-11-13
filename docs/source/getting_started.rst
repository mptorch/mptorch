Getting started
===============

Requirements
------------

- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, which installs everything else, Python and PyTorch included
- A C++ compiler (GCC >= 10 on Linux: the extension is compiled as C++20)
- CUDA >= 12.0, only if the CUDA kernels are wanted
- On macOS, nothing more. The Apple GPU (MPS) kernels are built with the rest and compiled for the GPU when first used, so Xcode is not needed.

MPTorch is a C++/CUDA extension, so installing it compiles the kernels against the PyTorch it is installed with.

Installation
------------

From the root of the repository:

.. code-block:: console

   $ uv sync

This downloads the Python that ``.python-version`` names (3.14), creates ``.venv`` with it, installs the packages pinned in ``uv.lock`` (PyTorch, and the tools for the tests, the linters, the documentation and the examples), and then compiles MPTorch against that PyTorch and installs it in editable mode. Commands run in the environment through ``uv run``:

.. code-block:: console

   $ uv run pytest tests/
   $ uv run python my_script.py

To update, pull and run ``uv sync`` again. ``uv run`` syncs too, before it runs its command. Either one rebuilds the extension when ``pyproject.toml``, ``setup.py`` or a file under ``mptorch/csrc/`` changed. The first build compiles about 180 source files and takes several minutes. The compiled objects are kept under ``build/``, so a rebuild compiles only the files the change reaches, and takes seconds when nothing changed. Delete ``build/`` to start from scratch. If `ccache <https://ccache.dev>`_ is installed, PyTorch's extension builder uses it without further setup, and a build from scratch then reuses whatever it has compiled before.

The CUDA kernels are built when ``torch.cuda.is_available()`` is true and ``CUDA_HOME`` is set. The Apple GPU kernels are built on macOS when PyTorch was built with MPS. Environment variables select other builds: ``USE_CUDA=0`` leaves out the CUDA kernels, ``USE_MPS=0`` the MPS ones, and ``DEBUG=1`` builds with debug symbols and no optimization. ``uv sync`` and ``uv run`` rebuild whenever one of them changes, so export it for the session rather than setting it on one command:

.. code-block:: console

   $ export USE_CUDA=0
   $ uv sync

A first example
---------------

Two objects carry most of the library: a *format*, which is a plain value, and a *quantizer*, which is that format turned into a function on tensors. Layers take one quantizer per signal, and anything left unset is plain PyTorch.

.. literalinclude:: ../snippets/getting_started.py
   :language: python
   :caption: docs/snippets/getting_started.py

.. literalinclude:: ../snippets/getting_started.out
   :language: text
   :caption: output

``BinaryK(8, 4)`` is ``Binary8p4se`` from IEEE P3109, the upcoming standard for machine-learning number formats: an 8-bit float with a sign bit, 4 exponent bits and 4 bits of precision (one of them implicit). The OCP format E4M3 has the same field widths, with an exponent bias one less. ``1.541`` becomes ``1.5`` because the representable values between 1 and 2 are ``1/8`` apart. :doc:`concepts` explains the formats and the rounding, and the guides build on it.

Vocabulary
----------

A few words are used throughout, with a fixed meaning:

format
   A description of which numbers can be represented: :class:`~mptorch.BinaryK` (the IEEE P3109 binary formats), :class:`~mptorch.SuperFP` or :class:`~mptorch.FixedPoint`, for example. Frozen, hashable, comparable by value.
rounding mode
   How a number that is *not* representable is mapped to one that is: :class:`~mptorch.RoundMode`. A property of an operation, not of a format.
quantize / quantizer
   To round every element of a tensor to a format; the function that does it. In MPTorch this is always "compute in the carrier, round the result".
carrier
   The IEEE 754 format a simulation computes and rounds in. By default it is binary32 for float32, float16 and bfloat16 tensors, and binary64 for float64 ones. It bounds which formats can be simulated in full. A carrier is named ``torch.float32`` or ``torch.float64`` and is never narrower than the dtype of the tensor it computes. The result keeps the tensor's dtype (see :doc:`concepts`).
signal
   One of the tensors flowing through a layer: its input, weight, bias, the gradient arriving from above, and the gradients it produces.
arithmetic
   What happens *inside* a matrix product: which format each partial product and each running sum is rounded to. A plain quantizer cannot express this; MPTorch's kernels exist for it.
MAC
   Multiply-accumulate: the arithmetic of one step of a dot product, as a value. It is a :class:`~mptorch.quant.SplitMac` (the product and the sum are rounded separately) or a :class:`~mptorch.quant.FusedMac` (one fused multiply-add, rounded once). An *accumulate algorithm* says in what order the terms are summed (see :doc:`kernels/arithmetic`).
block format
   Narrow element codes that share one scale per *block* of consecutive elements, or per *tile* of rows of blocks. :class:`~mptorch.BlockFormat` covers OCP's MX formats and NVIDIA's NVFP4 (see :ref:`block-formats`). A tensor in a block format is *packed* in the format's own bytes.
pass
   One of the matrix products of a training step: the forward product, and the two gradient products of the backward pass. Each runs in the arithmetic of its own hook, and each is a product of the one core routine (see :doc:`kernels/index`).
