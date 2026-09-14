MPTorch
=======

**MPTorch** is a PyTorch extension for *simulating* custom and mixed-precision
arithmetic in deep-learning workloads, in particular for training. Modern
accelerators compute in 8-bit floating point, in bfloat16, in formats that did
not exist a few years ago; hardware that does not exist yet will compute in
others. MPTorch lets you answer, on the CPU or GPU you already have, the
question every such format raises: *what happens to my model if this tensor,
or the arithmetic inside this matrix product, is rounded to that format?*

The simulation is exact rather than approximate: every operation is carried
out in IEEE binary32 -- or, for a float64 model, binary64 -- and its result is
rounded to the format you named, with the rounding mode you named, at the
points you named. A value that comes out of MPTorch is a value the simulated
hardware would have produced.

What you can do today
---------------------

- **Describe a floating-point format as a value.** :class:`~mptorch.BinaryK`
  is the binary format family of IEEE P3109, the upcoming standard for
  machine-learning arithmetic: any *K*-bit float with *P* bits of precision,
  signed or unsigned, with or without infinities. Setting its exponent bias
  and subnormal policy yourself reaches beyond the standard -- E4M3, E5M2,
  bfloat16, float16, or something nobody has built.
  :class:`~mptorch.SuperFP` is a second family that trades mantissa for
  dynamic range.
- **Round a tensor to it** with seven rounding modes, including stochastic
  rounding with a configurable number of random bits
  (:doc:`quantizers`).
- **Multiply matrices in it.** The arithmetic *inside* the dot product --
  each product, each addition to the running sum, or each fused
  multiply-add -- is rounded to a format you choose, on CPU and CUDA, with
  the full ``torch.matmul`` operand contract and gradients (:doc:`gemm`).
- **Vary the format per output element** of a matrix product, from a palette
  of up to eight formats.
- **Train with it.** :class:`~mptorch.quant.QLinear`,
  :class:`~mptorch.quant.QConv1d`/``2d``/``3d`` and
  :class:`~mptorch.quant.QMatmul` are drop-in layers whose every signal --
  input, weight, bias, and each gradient -- and whose every matrix product --
  forward, input gradient, weight gradient -- can be given its own format
  (:doc:`layers`). :class:`~mptorch.quant.Quantizer` is a straight-through
  estimator with one format forward and another backward.
- **Choose the arithmetic the simulation runs in.** A float64 model is
  computed in binary64 throughout, which simulates formats up to 53 bits of
  precision and ten exponent bits; ``carrier="binary32"`` runs the same model
  the float32 way, bit for bit (:doc:`concepts`).
- **Reproduce it.** ``torch.manual_seed`` controls the stochastic rounding
  streams, and the deterministic modes are bit-identical between CPU and
  CUDA.

Where to start
--------------

:doc:`getting_started` installs the extension and runs a first example.
:doc:`concepts` explains the formats and rounding modes, with the equations
the implementation follows. The three guides -- :doc:`quantizers`,
:doc:`gemm`, :doc:`layers` -- cover every exposed function and class with a
worked example each. :doc:`tutorial` puts it all together on the 8-bit
formats of current NVIDIA GPUs, up to training a network in them.

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

Every code block in these pages is a file under ``docs/snippets/``, and the
output shown after it is what that file printed when
``docs/run_snippets.py`` ran it. Nothing is transcribed by hand. The recorded
outputs come from a laptop with an NVIDIA RTX 4060 (Ada, CUDA 13.2), Python
3.13 and PyTorch 2.14; to regenerate them on your own machine:

.. code-block:: console

   $ python3 docs/run_snippets.py            # every snippet
   $ python3 docs/run_snippets.py rounding   # those whose name contains "rounding"

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
