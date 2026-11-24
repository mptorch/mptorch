MPTorch
=======

**MPTorch** is a PyTorch extension that simulates custom number formats and
mixed-precision arithmetic in deep-learning models, for training as well as
inference. Recent accelerators compute in narrow formats, such as 8-bit
floating point, or 4-bit formats whose values share one scale per block.
MPTorch lets you run a model in such formats on hardware that does not
support them: a CPU, an NVIDIA GPU or an Apple GPU.

It simulates two kinds of rounding:

- **Rounding a tensor** to a format, element by element: a layer's weights,
  its inputs, or the gradients of the backward pass.
- **Rounding inside a matrix product or a convolution**: each product and
  each partial sum of a dot product can be rounded to a format of your
  choice.

The values stay in an ordinary PyTorch dtype: float32, float64, float16 or
bfloat16. Each operation is computed in a wider IEEE 754 format, the
*carrier* (binary32 or binary64), and its result is then rounded to the
target format. In most cases the result is exactly what hardware that
computes in the target format would return (:doc:`concepts`, "The simulation
model").

Features
---------------------

- **Floating-point formats.** :class:`~mptorch.BinaryK` is the family of
  binary floating-point formats of IEEE P3109, the upcoming standard for
  machine-learning arithmetic: any *K*-bit float with *P* bits of precision,
  signed or unsigned, with or without infinities. MPTorch goes beyond the
  standard with custom exponent biases and two other ways of handling the
  range below the smallest normal number. :class:`~mptorch.SuperFP` is a
  second family: it keeps a full mantissa only in its top binades, and
  spends the remaining codes on powers of two below them, which gives it a
  much wider range for the same number of bits.
- **Fixed-point formats.** :class:`~mptorch.FixedPoint` has any word length
  and any number of fractional bits, in two's complement, symmetric or
  unsigned. Its elementwise quantizer supports every rounding mode (see
  :ref:`fixed-point`).
- **Block formats.** In a block format, each block of consecutive elements
  shares one scale. :class:`~mptorch.BlockFormat` covers the OCP
  Microscaling (MX) formats, NVIDIA's NVFP4 and block floating point. A
  scale can cover a 1D block or a 2D tile. The elements and the scales can
  be BinaryK or SuperFP formats, and the elements can also be fixed point.
  You also choose how each block's scale is rounded (see
  :ref:`block-formats`).
- **Quantizers**: functions that round a tensor to a format, in nine
  rounding modes. Three of the modes are stochastic, with a configurable
  number of random bits. Block quantizers can also store a tensor packed in
  the block format's own bytes (see :doc:`quantizers`).
- **Custom arithmetic kernels.** Inside a dot product, you choose the format
  that each product, each addition to the running sum, or each fused
  multiply-add is rounded to, and the order in which the terms are summed.
  One matrix-product routine computes all of these, on the CPU, on CUDA
  GPUs and on Apple GPUs. It runs:

  - matrix products, with every operand shape that ``torch.matmul`` accepts,
    and their gradients;
  - convolutions, without unfolding their inputs;
  - products of block-format tensors, which are decoded as they are read.

  A *palette* of up to eight formats lets each output element of one matrix
  product use its own format (see :doc:`kernels/index`).
- **Layers for mixed-precision training.** :class:`~mptorch.quant.QLinear`
  and :class:`~mptorch.quant.QConv1d`/``2d``/``3d`` are subclasses of
  ``nn.Linear`` and ``nn.Conv1d``/``2d``/``3d`` that take one more argument,
  ``formats``. :class:`~mptorch.quant.QMatmul` is the same for
  ``torch.matmul``. Each tensor that flows through a layer (its input,
  weight and bias, and each gradient) can be rounded to its own format, and
  each of its matrix products (forward, input gradient, weight gradient) can
  have its own arithmetic (see :doc:`layers`). To round a tensor between
  layers, :class:`~mptorch.quant.Quantizer` uses one format in the forward
  pass and another for the gradient.
- **Existing models.** A *recipe* gives a model's products, activations,
  norms, attention and arithmetic their formats, by kind and by module.
  :func:`~mptorch.quant.convert` switches the model's layers to mptorch's;
  :class:`~mptorch.quant.simulate` runs the model as it is, function calls
  included, such as a transformer's attention; and
  :func:`~mptorch.quant.profile` counts what it computes, by format, in
  both passes (see :doc:`instrument`).
- **Wide formats.** The binary64 carrier simulates formats of up to 53 bits
  of precision and ten exponent bits. A float64 model computes in binary64.
  ``carrier=torch.float64`` gives a float32, float16 or bfloat16 model the
  same arithmetic, while its values keep their own dtype (see
  :doc:`concepts`).
- **Reproducibility.** ``torch.manual_seed`` makes stochastic rounding
  reproducible. In the deterministic rounding modes, the CPU and CUDA
  kernels return bit-identical results. On an Apple GPU every result is the
  CPU's, stochastic rounding included, except where the GPU flushes a
  binary32 subnormal number to zero inside a matrix product (see
  :ref:`apple-gpu`).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   concepts
   quantizers
   kernels/index
   layers
   instrument
   tutorial
   api

About the examples
------------------

Every code block in these pages is a file under ``docs/snippets/``. The
output shown after it is what that file printed when ``docs/run_snippets.py``
ran it. To regenerate the outputs on your own machine:

.. code-block:: console

   $ uv run python docs/run_snippets.py            # every snippet
   $ uv run python docs/run_snippets.py rounding   # those whose name contains "rounding"

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
