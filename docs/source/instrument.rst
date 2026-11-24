Instrumenting a model
=====================

A model that already exists, such as a torchvision CNN or a Hugging Face
transformer, need not be rebuilt from mptorch's layers to compute in
mptorch's arithmetic. A *recipe* says what each part of the model computes
in, and three functions apply it:

- :func:`~mptorch.quant.convert` (or :func:`~mptorch.quant.convert_`, in
  place) switches the model's layers to mptorch's, which keep their
  parameters, so the converted model saves, loads and trains as the float
  one did;
- :class:`~mptorch.quant.simulate` runs the model as it is and intercepts
  its calls while it is active, so it also reaches what no layer computes:
  a product computed by a function call, attention, a model's own
  arithmetic;
- :func:`~mptorch.quant.profile` counts the model's products, sums and
  elements by the formats they compute in, and prices them.

On what both reach, a converted model and a simulated one compute the same
words, gradients and stochastic rounding included. The layers themselves
are :doc:`layers`'s, and what each one computes is described in
:doc:`kernels/index`.

Recipes
-------

A :class:`~mptorch.quant.Recipe` gives formats to *kinds* of computation,
and ``override(where, ...)`` changes them for the modules that ``where``
chooses: a glob over their names (``"encoder.*"``), a type, or a predicate.
The last rule that matches wins, and ``None`` keeps a module PyTorch's.

- :class:`~mptorch.quant.GemmFormats` holds one product's formats: the
  operand and gradient quantizers, a rounding of the output, and one mac
  for each pass (the forward product, the input gradient and the weight
  gradient), any of them ``None`` for PyTorch's own arithmetic.
  ``GemmFormats.uniform(mac, quant)`` fills them all. It is the value of
  the ``linear``, ``conv`` and ``matmul`` kinds, and ``gemm`` names the
  three.
- The other kinds each take the formats of their operations:
  ``activation`` an :class:`~mptorch.quant.ActivationFormats`,
  ``embedding`` and ``elementwise`` a
  :class:`~mptorch.quant.PointwiseFormats`, ``softmax`` a
  :class:`~mptorch.quant.SoftmaxFormats`, ``norm`` a
  :class:`~mptorch.quant.NormFormats`, ``reduction`` a
  :class:`~mptorch.quant.ReductionFormats` and ``attention`` an
  :class:`~mptorch.quant.AttentionFormats`. A recipe that names none of
  them leaves those operations alone, and reports nothing about them.
- ``mptorch.quant.presets`` holds three recipes: ``uniform(fmt)``,
  ``fp8()`` (E4M3 operands and E5M2 gradients, each scaled per tensor) and
  ``mx()`` (OCP MX block formats).

``recipe.plan(model)`` shows what a conversion would do. Each kind reaches
these calls:

.. list-table::
   :header-rows: 1
   :widths: 14 38 48

   * - kind
     - a conversion converts
     - a simulation also reaches
   * - ``linear``
     - ``nn.Linear``; ``nn.MultiheadAttention``'s projections
     - ``F.linear``, and a ``torch.addmm`` or ``torch.matmul`` whose right
       operand is one of the model's weights (GPT-2's ``Conv1D``)
   * - ``conv``
     - ``nn.Conv1d``/``2d``/``3d``
     - ``F.conv1d``/``2d``/``3d``
   * - ``matmul``
     - a class registered for it
     - a product of two activations, at its site
   * - ``activation``
     - torch's activation modules
     - ``F.gelu``, ``F.silu``, ``F.hardswish``, ``F.hardsigmoid``,
       ``F.relu``, ``torch.sigmoid``, ``torch.tanh``
   * - ``embedding``
     - ``nn.Embedding``
     - ``F.embedding``
   * - ``elementwise``
     - (no module)
     - ``+``, ``-``, ``*``, ``/``, ``sqrt``, ``rsqrt``, ``exp``, a power
   * - ``softmax``
     - ``nn.Softmax``
     - ``F.softmax``, ``torch.softmax``, ``Tensor.softmax``
   * - ``norm``
     - torch's norm modules; a class registered for it
     - ``F.layer_norm``, ``F.rms_norm``, ``F.group_norm``, ``F.batch_norm``
   * - ``reduction``
     - (no module)
     - ``torch.sum``, ``torch.mean``
   * - ``attention``
     - ``nn.MultiheadAttention``'s core
     - ``F.scaled_dot_product_attention``

Converting a model
------------------

:func:`~mptorch.quant.convert` converts a copy of the model, and
:func:`~mptorch.quant.convert_` the model itself, in place. Each module of
a class in the table that the recipe gives formats to becomes its mptorch
twin: an ``nn.Linear`` a :class:`~mptorch.quant.QLinear`, an
``nn.Conv2d`` a :class:`~mptorch.quant.QConv2d`, an ``nn.Softmax`` a
:class:`~mptorch.quant.QSoftmax`, an ``nn.MultiheadAttention`` a
:class:`~mptorch.quant.QMultiheadAttention`, and so on (the modules of a
kind past the GEMMs only when the recipe names it). The module is the same
object with its class changed, so its parameters, buffers, hooks and
``state_dict`` keys stay as they were, and an optimizer built before the
conversion still trains it. Only those exact classes convert. A subclass,
a parametrized module among them, is listed and kept, and ``strict=True``
raises instead. :func:`~mptorch.quant.unconvert_` turns the layers back,
and :func:`~mptorch.quant.register` adds a class of your own.

.. literalinclude:: ../snippets/instrument_convert.py
   :language: python
   :caption: docs/snippets/instrument_convert.py

.. literalinclude:: ../snippets/instrument_convert.out
   :language: text
   :caption: output

A converted model is saved and loaded through its ``state_dict``, which is
the float model's: the layers' formats hold closures, which the recipe
rebuilds. Only modules convert, so a product that a model's code computes
with a function, such as a ``torch.matmul`` in an attention block, keeps
PyTorch's arithmetic.

Simulating a model
------------------

:class:`~mptorch.quant.simulate` runs a model in a recipe's arithmetic
without converting it, and reaches what a conversion cannot. While it is
active (``with simulate(recipe, model) as sim:``), it intercepts the model's
calls: each call the recipe gives formats to runs through the autograd
Function a converted layer runs, with the same formats, so on the layers a
conversion would convert the results are the converted model's, word for
word, gradients included. A product is resolved by kind:

- ``linear`` is ``F.linear``, and a ``torch.addmm`` or ``torch.matmul``
  whose right operand is one of the model's Parameters (GPT-2's ``Conv1D``
  is an ``addmm``). It takes the formats of the module that owns the
  weight, so ``F.linear(x, self.qkv.weight)`` takes ``qkv``'s even when
  ``qkv`` itself is never called;
- ``conv`` is ``F.conv1d/2d/3d``, resolved the same way;
- ``matmul`` is a product of two activations (a view of a weight, such as
  ``W.T``, counts as one). It takes the formats of the module that computes
  it, at a *site*: ``"matmul#0"`` is that module's first such product in a
  forward pass and ``"matmul#1"`` its second, so ``override(where,
  site="matmul#1", matmul=...)`` reaches one of them. The report lists the
  sites.

``sim.report()`` lists every call seen, what ran it, and why a call was
left to PyTorch: the recipe gave it ``None``, or simulate cannot run it (an
``einsum``; ``F.scaled_dot_product_attention``, which computes an attention
block's products in one call, under a recipe with no ``attention`` kind),
and ``strict=True`` raises for the latter in a module the recipe covers. A
converted model's layers run as converted. Under ``torch.compile``, compile
``sim.wrap()``, which enters the simulation inside the compiled code; a
compiled call is not recorded in the report.

.. literalinclude:: ../snippets/instrument_simulate.py
   :language: python
   :caption: docs/snippets/instrument_simulate.py

.. literalinclude:: ../snippets/instrument_simulate.out
   :language: text
   :caption: output

Activations, embeddings and a model's own arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A recipe that names the ``activation``, ``embedding`` or ``elementwise``
kind reaches more than the products, and only then:

- ``activation`` is ``F.gelu``, ``F.silu``, ``F.hardswish``,
  ``F.hardsigmoid``, ``F.relu``, ``torch.sigmoid`` and ``torch.tanh``, under
  the module that calls them, so Hugging Face's ``SiLUActivation`` and a
  torch ``nn.GELU`` alike. A call with ``inplace=True`` computes out of
  place, as the mptorch modules do;
- ``embedding`` is ``F.embedding``, under the module that owns the weight;
- ``elementwise`` is a model's own arithmetic: ``+``, ``-``, ``*``, ``/``
  and their ``torch.`` spellings, ``sqrt``, ``rsqrt``, ``exp`` and a power
  with a number exponent, each through its op of
  :doc:`kernels/pointwise`, under the module whose code computes it.
  Residual sums, attention's scaling and mask, RMSNorm's ``rsqrt`` and
  RoPE's products are all of this kind, so a recipe usually names it for
  the blocks only (``override("*.layers.*", elementwise=...)``): an additive
  mask rounded to an 8-bit format, for instance, becomes its infinities.
  Arithmetic on integer tensors is not rounded, and neither are the
  in-place forms (``x += y``), which the report lists.

.. literalinclude:: ../snippets/instrument_activations.py
   :language: python
   :caption: docs/snippets/instrument_activations.py

.. literalinclude:: ../snippets/instrument_activations.out
   :language: text
   :caption: output

Softmax, norms and sums
~~~~~~~~~~~~~~~~~~~~~~~

The ``softmax`` and ``norm`` kinds give their formats to torch's modules,
converted to :doc:`layers`'s twins, and, for a simulation, to the
functions too:

- ``softmax`` is ``nn.Softmax`` for a conversion, and ``F.softmax``,
  ``torch.softmax`` and ``Tensor.softmax`` for a simulation, which also
  reaches the softmax an attention block calls as a function;
- ``norm`` is torch's norm modules for a conversion, and ``F.layer_norm``,
  ``F.rms_norm``, ``F.group_norm`` and ``F.batch_norm`` for a simulation;
- ``reduction``, for a simulation only, is a model's ``torch.sum`` and
  ``torch.mean`` over float tensors, through :func:`~mptorch.quant.qsum` and
  :func:`~mptorch.quant.qmean`. A sum of integers, such as a mask's count,
  is left alone.

A norm that a model writes out of elementwise ops, such as Hugging Face's
``LlamaRMSNorm``, is no torch class, and no functional call reveals it.
Registering its class with the forward it should compute,
``register(cls, kind="norm", forward=...)``, reaches it as a whole: a
conversion switches it to a class that runs that forward, and a simulation
runs that forward in place of the module's own, whose calls (its children's
included) it then passes through. Without the registration, its ``mean`` is
a ``reduction`` and its other steps ``elementwise``, each rounded on its
own.

.. literalinclude:: ../snippets/instrument_norms.py
   :language: python
   :caption: docs/snippets/instrument_norms.py

.. literalinclude:: ../snippets/instrument_norms.out
   :language: text
   :caption: output

Attention
~~~~~~~~~

A recipe reaches a model's attention by kind:

- ``attention`` is ``nn.MultiheadAttention``'s core for a conversion, and
  for a simulation also ``F.scaled_dot_product_attention`` wherever a model
  calls it (transformers' default), through
  :func:`~mptorch.quant.qattention`. Without the kind both are reported as
  not covered.
- ``linear`` gives ``nn.MultiheadAttention``'s projections their formats, as
  for any layer with a weight: a recipe for the GEMMs converts the module
  for them, its core left PyTorch's. Its ``out_proj`` is the module's own
  and converts with it.

An eager attention, written out of two ``torch.matmul`` calls and a
softmax, has no single call to reach. ``recipe.attention_parts(where,
formats)`` gives one ``AttentionFormats`` to both spellings: the
``attention`` kind, and the eager parts, ``formats.qk`` at site
``"matmul#0"``, ``formats.softmax`` and ``formats.pv`` at ``"matmul#1"``.
An SDPA model and its eager twin then compute the same words, gradients
included (:func:`~mptorch.quant.qattention` lists the cases where they
cannot).

.. literalinclude:: ../snippets/instrument_attention.py
   :language: python
   :caption: docs/snippets/instrument_attention.py

.. literalinclude:: ../snippets/instrument_attention.out
   :language: text
   :caption: output

Counting a model's arithmetic
-----------------------------

:func:`~mptorch.quant.profile` runs one forward pass of a model and counts
what it computes, by the formats each part computes in: the model as it
is, its converted layers by their own formats, or, with ``recipe=``, as
``simulate(recipe, model)`` would run it. A
:class:`~mptorch.quant.Profile` holds three lists:

- ``records``, one :class:`~mptorch.quant.ProfileRecord` per product and
  per sum, as the GEMM the kernels run: each product (a linear layer, a
  convolution, a product of two activations, an attention's scores and
  output, a projection of ``nn.MultiheadAttention``), and each sum of a
  softmax, a norm or a reduction, which mptorch runs as a product with a
  vector of ones. A record is filed under its mac, the *key*, which a cost
  table prices per MAC (``{BinaryK(8, 4): 1.0, None: 4.0}``, a mac priced
  by its product format, ``None`` for PyTorch's arithmetic);
- ``elements``, one :class:`~mptorch.quant.ElementRecord` per elementwise
  call (an activation, an arithmetic op, an embedding, a softmax's or a
  norm's steps), with the number of elements it computes. The cost table
  does not price them;
- ``uncounted``, the calls it saw and could not count (an ``einsum``, a
  module simulated whole).

``backward=True`` adds the backward pass's products and sums, from the
forward's shapes, each under its own pass's mac (``igrad_mac``,
``wgrad_mac``, a sum's ``bwd_acc``), for each operand that requires a
gradient: not the data's, not a frozen weight's. ``run=False`` runs the
pass under a fake tensor mode, which computes and allocates nothing, so a
model too large to run, built on the meta device, is counted from its
shapes alone.

.. literalinclude:: ../snippets/instrument_profile.py
   :language: python
   :caption: docs/snippets/instrument_profile.py

.. literalinclude:: ../snippets/instrument_profile.out
   :language: text
   :caption: output

The counts are of the model's arithmetic as mptorch's kernels run it: a
convolution's input gradient is one product per residue class of the
stride, over the kernel taps that reach its positions (the same MACs as the
forward pass for a padded convolution, a few more for an unpadded one), and
an attention's grouped keys and values count once per query head, which
they are repeated for. The pass leaves the model's buffers (a batch norm's
running statistics) and the random generators as it found them.

Three models
------------

A torchvision ResNet-18, by conversion: every convolution and the
classifier but the last, in an 8-bit recipe, then the forward and backward
pass's MACs, priced.

.. literalinclude:: ../snippets/instrument_resnet.py
   :language: python
   :caption: docs/snippets/instrument_resnet.py

.. literalinclude:: ../snippets/instrument_resnet.out
   :language: text
   :caption: output

A torchvision ViT-B/16, by simulation: ``presets.fp8()`` for the products,
and the ``attention`` kind for the encoder's attention, whose
``nn.MultiheadAttention`` the simulation reaches whole.

.. literalinclude:: ../snippets/instrument_vit.py
   :language: python
   :caption: docs/snippets/instrument_vit.py

.. literalinclude:: ../snippets/instrument_vit.out
   :language: text
   :caption: output

A model written as transformers writes Llama: its RMSNorm out of
elementwise ops, registered as a whole; SDPA with grouped keys
(``enable_gqa=True``); a gated MLP. The blocks' norms, activations and
arithmetic are in a bfloat16-precision format, the products in 8 bits, the
classifier PyTorch's.

.. literalinclude:: ../snippets/instrument_llama.py
   :language: python
   :caption: docs/snippets/instrument_llama.py

.. literalinclude:: ../snippets/instrument_llama.out
   :language: text
   :caption: output

What is and is not covered
--------------------------

- **A conversion** reaches the modules of the registered classes, and
  nothing a model computes with a function.
- **A simulation** reaches every call in the kinds' table, under the
  modules of the model it was made for. Calls made outside those modules
  are not intercepted, and neither are those inside an mptorch layer (a
  converted model's layers run as converted) or inside a module simulated
  whole.
- **Neither** reaches the products ``simulate`` does not run (``einsum``,
  ``tensordot``, ``F.conv_transpose*d``, a ``matmul`` with ``out=``), which
  the report lists, nor mptorch's own functions (:func:`~mptorch.quant.qmatmul`,
  :func:`~mptorch.quant.qattention`, ...) called in a model's code, whose
  bodies hide their calls.
- **A profile** counts what a simulation reaches, and lists the rest: the
  products ``simulate`` does not run and the arithmetic of a module
  simulated whole, whose elements it counts.

Measured on pretrained models
-----------------------------

``examples/instrument/`` runs the instrumentation on pretrained weights
(an RTX 4060 laptop GPU, 2026-11-24). ``vision.py`` scores torchvision's
ResNet-18, converted, and ViT-B/16, simulated, on 1,000 images of
Imagenette's validation set, top-1 over ImageNet's 1,000 classes, in four
recipes: the model as it is; ``binary8``, every product's operands in
Binary8p4, its products rounded to it and summed in Binary16p11;
``presets.fp8()``; and ``presets.mx()``, which converts the linear layers
only (ResNet-18's classifier, 0.03% of its MACs). ViT-B/16's attention
takes each recipe's arithmetic too.

==========  ===============  ===============
recipe      ResNet-18        ViT-B/16
==========  ===============  ===============
float32     77.5%, 681/s     90.9%, 158/s
binary8     75.8%, 61/s      91.7%, 6.6/s
fp8         77.6%, 174/s     91.3%, 21.6/s
mx          79.1%, 867/s     90.7%, 18.5/s
==========  ===============  ===============

Each rate is images per second, float32's the data loader's. Over 1,000
images, a top-1 accuracy near 80% is known to within about 1.3 points (one
standard deviation), so only ResNet-18's ``binary8`` stands apart.

``gpt2_perplexity.py`` scores Hugging Face's own pretrained GPT-2 small,
through ``simulate``, on the first 32,768 tokens of WikiText-2's test split
(windows of 1,024 tokens moved by 512): ``presets.fp8()`` on every
projection but the tied ``lm_head``; then the attention's two products on
E4M3 operands too, and its softmax in Binary14p8, bfloat16's precision;
then the blocks' own arithmetic (the residual sums, the GELU's steps) and
their norms in Binary14p8 as well.

==================  ==========  ===========  ==========================
recipe              perplexity  tokens/s     MACs in the recipe's formats
==================  ==========  ===========  ==========================
float32             26.76       20,076
fp8                 27.65       3,968        59.6%
fp8 and attention   28.59       2,870        72.9%
and the blocks      32.61       2,742        72.9%
==================  ==========  ===========  ==========================

The SDPA model and its eager twin (``attn_implementation="eager"``) under
``attention_parts`` compute the same logits, word for word, and under the
last recipe nothing the model calls is left uncovered. The rest of the
MACs are ``lm_head``'s, a third of a GPT-2 small's. Rounding the blocks'
residual sums, GELUs and norms to bfloat16's precision costs more
perplexity than all their products in FP8.
``dev/benchmarks/timing/gpt2_simulated.py`` measures what simulating it
costs: on CUDA the ``fp8`` recipe with the attention kind takes 7.4 times
float32's time, 70% of it in mptorch's GEMMs, and
``torch.compile(sim.wrap())`` compiles the whole model into one graph in
30 s and runs it 1.15 times faster.
