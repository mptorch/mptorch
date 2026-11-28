Getting started
===============

Requirements
------------

MPTorch is a PyTorch extension with compiled kernels, so installing it
compiles them against the PyTorch it is installed with. It needs:

- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, which
  installs everything else, Python and PyTorch included. MPTorch supports
  Python 3.14 or later, with any PyTorch release that has wheels for it.
- A C++20 compiler: GCC 10 or later on Linux, Apple clang on macOS.
- For the CUDA kernels only: an NVIDIA GPU, visible when MPTorch is built,
  and the CUDA toolkit, version 12.0 or later.
- On macOS, nothing more. The Apple GPU (MPS) kernels are built with the
  rest and compiled for the GPU when first used, so Xcode is not needed.

Installation
------------

From the root of the repository:

.. code-block:: console

   $ uv sync

This command:

1. downloads the Python version that ``.python-version`` names (3.14);
2. creates the environment ``.venv`` with it;
3. installs the packages pinned in ``uv.lock``: PyTorch, and the tools for
   the tests, the linters, the documentation and the examples;
4. compiles MPTorch against that PyTorch and installs it in editable mode,
   so that edits to its Python files take effect without reinstalling.

Run commands in the environment through ``uv run``:

.. code-block:: console

   $ uv run pytest tests/
   $ uv run python my_script.py

To update, pull and run ``uv sync`` again. ``uv run`` also syncs before it
runs its command. Either one rebuilds the extension when ``pyproject.toml``,
``setup.py`` or a file under ``mptorch/csrc/`` has changed.

The first build compiles 235 source files with the CUDA kernels, or 122
without them (136 on macOS, with the Apple GPU kernels), and takes several
minutes. The compiled objects are kept under ``build/``, so a rebuild
compiles only the files that the change affects, and takes seconds when
nothing changed. Delete ``build/`` to start from scratch. If `ccache
<https://ccache.dev>`_ is installed, PyTorch's extension builder uses it
without further setup. A build from scratch then reuses every object that
ccache has compiled before.

The CUDA kernels are built when ``torch.cuda.is_available()`` is true and
``CUDA_HOME`` is set. Otherwise the build leaves them out, without an error.
The Apple GPU kernels are built on macOS when PyTorch was built with MPS.

Environment variables select other builds:

- ``USE_CUDA=0`` leaves out the CUDA kernels.
- ``USE_MPS=0`` leaves out the Apple GPU kernels.
- ``MPTORCH_NO_ODD=1`` leaves out every kernel that rounds the carrier's
  steps to odd, for a faster build. These kernels run under
  ``carrier_rounding=RoundMode.RO``, which keeps the carrier's own rounding
  of each product and sum from changing the format's rounding
  (:doc:`kernels/arithmetic`, "What the carrier rounds"). Without them,
  those calls raise an error that names the variable.
- ``MPTORCH_ODD_ACCUMULATED=1`` adds the kernels that round the carrier's
  steps to odd under the ``KAHAN``, ``BLOCK`` and ``TREE`` accumulate
  algorithms. They are left out by default because they add about 44% to
  the build's compile work. Without them, those calls raise an error that
  names the variable. ``MPTORCH_NO_ODD=1`` leaves them out in any case.
- ``MPTORCH_NO_FP64=1`` leaves out the binary64 kernels of the matrix
  products, the convolutions and the block formats, for a faster build.
  Those ops then refuse float64 tensors with an error that names the
  variable. The elementwise quantizers keep their binary64 path.
- ``DEBUG=1`` builds with debug symbols and no optimization.

``uv sync`` and ``uv run`` rebuild whenever one of these variables changes.
Export a variable for the whole session, rather than setting it on one
command, or the next command without it rebuilds again:

.. code-block:: console

   $ export USE_CUDA=0
   $ uv sync

A first example
---------------

Two objects carry most of the library. A *format* is a plain value that
describes a set of numbers. A *quantizer* is that format turned into a
function on tensors. A layer takes one quantizer per signal (each tensor that
flows through it), and anything left unset is plain PyTorch.

.. literalinclude:: ../snippets/getting_started.py
   :language: python
   :caption: docs/snippets/getting_started.py

.. literalinclude:: ../snippets/getting_started.out
   :language: text
   :caption: output

``BinaryK(8, 4)`` is ``Binary8p4se`` from IEEE P3109, the upcoming standard
for machine-learning number formats. It is an 8-bit float with a sign bit,
4 exponent bits and 4 bits of precision, one of them implicit. The OCP
format E4M3 has the same field widths, with an exponent bias one less.
``1.541`` becomes ``1.5`` because the representable values between 1 and 2
are ``1/8`` apart. :doc:`concepts` explains the formats and the rounding, and
the other guides build on it.

A model you already have
------------------------

A model built from torch's modules computes in mptorch's arithmetic without
being rebuilt: a *recipe* says which formats each part of it computes in,
:func:`~mptorch.quant.convert` switches its layers to mptorch's, and
:func:`~mptorch.quant.profile` counts its multiply-accumulates (MACs) by
format.

.. literalinclude:: ../snippets/getting_started_model.py
   :language: python
   :caption: docs/snippets/getting_started_model.py

.. literalinclude:: ../snippets/getting_started_model.out
   :language: text
   :caption: output

:doc:`instrument` goes further: a model's attention, activations and norms,
the calls no layer makes, and the backward pass's cost.

Vocabulary
----------

A few words are used throughout, with a fixed meaning:

format
   A description of which numbers can be represented: for example
   :class:`~mptorch.BinaryK` (the IEEE P3109 binary formats),
   :class:`~mptorch.SuperFP` or :class:`~mptorch.FixedPoint`. A format
   object is frozen (it cannot be modified), hashable and compared by value.
rounding mode
   How a number that is *not* representable is mapped to one that is:
   :class:`~mptorch.RoundMode`. It is a property of an operation, not of a
   format.
quantize / quantizer
   To round every element of a tensor to a format; the function that does
   it. In MPTorch the value is always computed in the carrier first, and the
   result is then rounded.
carrier
   The IEEE 754 format that a simulation computes and rounds in. By default
   it is binary32 for float32, float16 and bfloat16 tensors, and binary64
   for float64 tensors. It bounds which formats can be simulated in full. A
   carrier is named ``torch.float32`` or ``torch.float64``, and it is never
   narrower than the dtype of the tensors it computes on. The result keeps
   the tensors' dtype (see :doc:`concepts`).
signal
   One of the tensors that flow through a layer: its input, weight and bias,
   the gradient arriving from the layer above, and the gradients it
   produces.
arithmetic
   What happens *inside* a matrix product: which format each product and
   each running sum is rounded to. A plain quantizer cannot express this,
   and MPTorch's kernels exist to simulate it.
MAC
   Multiply-accumulate: the arithmetic of one step of a dot product,
   described as a value. It is a :class:`~mptorch.quant.SplitMac` (the
   product and the sum are rounded separately) or a
   :class:`~mptorch.quant.FusedMac` (one fused multiply-add, rounded once).
   An *accumulate algorithm* says in what order the terms are summed (see
   :doc:`kernels/arithmetic`).
block format
   A format of narrow element codes that share one scale per *block* of
   consecutive elements, or per *tile* of several rows of blocks.
   :class:`~mptorch.BlockFormat` covers OCP's MX formats and NVIDIA's NVFP4
   (see :ref:`block-formats`). A tensor in a block format is *packed*: it is
   stored in the format's own bytes.
pass
   One of the matrix products of a training step: the forward product, and
   the two gradient products of the backward pass. Each pass runs in the
   arithmetic of its own hook, and each is computed by the one core
   routine (see :doc:`kernels/index`).
