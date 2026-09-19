MPTorch
=======

**MPTorch** is a PyTorch extension for *simulating* custom and mixed-precision arithmetic in deep-learning workloads, in particular for training. Modern accelerators increasingly use small wordlength formats (e.g. 8-bit floating-point, or 4-bit block-based formats) that were not available just a few years ago.

Every elementary operation can be simulated in a custom arithmetic. The operands are stored in a *high* precision payload (which can be either bfloat16, or the IEEE 754 binary16, binary32, or binary64 formats), and the operation is carried out in a *carrier* format, either IEEE binary32 or binary64. The result is rounded to the desired target format. In most cases, this simulated behavior is identical with what would be obtained by actually performing the operation in the target arithmetic.

Features
---------------------

- **Custom floating-point format types.** :class:`~mptorch.BinaryK` included the family of binary floating-point formats specified by IEEE P3109, the upcoming standard for machine-learning arithmetic: any *K*-bit float with *P* bits of precision, signed or unsigned, with or without infinities. It goes beyond the standard by allowing custom bias terms and different subnormal policies. :class:`~mptorch.SuperFP` is a second family that trades mantissa precision in certain portions of the representation domain for dynamic range.
- **Tensor rounding functions.** with seven rounding modes, including stochastic rounding with a configurable number of random bits (:doc:`quantizers`).
- **Custom arithmetic matrix multiplication.** The arithmetic *inside* the dot product -- each product, each addition to the running sum, or each fused multiply-add -- is rounded to a format you choose, on CPU, CUDA and Apple GPUs, with the full ``torch.matmul`` operand contract and gradients (:doc:`gemm`).
- **Mixed precision operations** in dot product chains of a matrix product, from a user-defined palette of up to eight formats.
- **Custom and mixed precision training support.** :class:`~mptorch.quant.QLinear`, :class:`~mptorch.quant.QConv1d`/``2d``/``3d`` and :class:`~mptorch.quant.QMatmul` are drop-in layers whose every signal -- input, weight, bias, and each gradient -- and whose every matrix product -- forward, input gradient, weight gradient -- can be given its own format (:doc:`layers`). :class:`~mptorch.quant.Quantizer` is a straight-through estimator with one format forward and another backward.
- **Configurable formats for simulating custom floating-point arithmetic.** An underlying binary64 arithmetic simulates formats up to 53 bits of precision and ten exponent bits; ``carrier=torch.float64`` gives binary32, binary16 or bfloat16 models and values the same arithmetic, while the values are kept in the target dtype (:doc:`concepts`).
- **Reproducibility.** ``torch.manual_seed`` controls the stochastic rounding streams, and the deterministic modes are bit-identical between CPU and CUDA. On an Apple GPU every result is the CPU's, stochastic rounding included, except where the GPU flushes a binary32 subnormal (:ref:`apple-gpu`).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   concepts
   quantizers
   gemm
   layers
   tutorial
   api

About the examples
------------------

Every code block in these pages is a file under ``docs/snippets/``, and the output shown after it is what that file printed when ``docs/run_snippets.py`` ran it. To regenerate them on your own machine:

.. code-block:: console

   $ python3 docs/run_snippets.py            # every snippet
   $ python3 docs/run_snippets.py rounding   # those whose name contains "rounding"

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
