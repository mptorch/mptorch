"""What the layers' autograd Functions keep for the backward pass.

``CustomArithLinear``, ``CustomArithConvNd`` and ``CustomArithMatmul`` need
each quantized operand again in backward, one per gradient: the quantized
input (or ``a``) for the weight's (``b``'s) gradient, and the quantized
weight for the input's. For each operand there are two choices:

* *keep the copy*: save the quantized tensor, a new tensor the size of the
  operand, from the forward pass;
* *recompute*: save the raw operand, and quantize it again in backward.

Recomputing costs one more quantize pass and saves the copy's memory. The
saving is real only where the raw operand stays alive anyway: a weight
always does, as the layer's parameter, and an input does when the op that
produced it keeps its own output for its backward (ReLU, softmax, sigmoid),
or when it is the data batch. The module of each Function says what a
``None`` flag means for it (``_LINEAR_RECOMPUTE_DEFAULT`` and its two
counterparts).

A quantizer is run a second time only if it is known to depend on nothing
but its input and the random generators: mptorch's ``Quant`` (when not in
place), ``ScaledQuant``, ``BlockQuant`` called as a fake quantizer, a
``Quantizer`` over any of them, and a ``GemmFormats`` rotation over any of
them or over nothing. The caller can set the flag to ``True`` to recompute with
any other callable, which is their promise that it too depends on nothing
else. Without that, any other callable keeps its result saved: a QAT
observer called twice would count a batch twice, or round differently in
backward once its scale had moved. A quantizer whose result is not a tensor
(a ``BlockPacked``) also keeps its result saved, since it is already
compact.

Under stochastic rounding the second call must draw the same random bits as
the first. So the state of the generators the quantizer reads is captured
just before the forward's call, set back for the backward's call, and then
restored to what backward found, so that the gradient quantizers and the
math hooks draw exactly what they drew before. mptorch's CUDA kernels draw
from the generator of their tensor's device (every CUDA op runs on that
device, ``csrc/cuda/device_guard.h``), and its CPU and MPS kernels from the
CPU generator (``csrc/cpu/utils.h``, and ``draw_cpu_seed`` on MPS). For a
caller's own callable, both the CPU generator and its tensor's device
generator are captured.

Under ``torch.compile``, a recompute that needs generator state keeps the
quantized copy instead: Dynamo cannot trace reading or setting a generator's
state, so the recompute would break the graph, and the copy holds the same
values the recompute would return. A recompute that needs no generator state
(a deterministic ``Quant``) is traced and runs as in eager mode.
"""

from collections.abc import Callable
from typing import Any, cast

import torch

from mptorch.number import RoundMode

from .. import events
from ..events import _CountingQuant
from ..mac import BlockQuant, Quant
from ..ops import BlockPacked
from ..rotation import _RotatedQuant
from ..scaled import ScaledQuant
from .quantizer import Quantizer

# (the CPU generator's state or None, ((device, that device's generator state), ...))
_State = tuple[torch.Tensor | None, tuple[tuple[torch.device, torch.Tensor], ...]]
# what backward needs to quantize a raw operand again: the quantizer, and the
# generator states to run it under (None when it needs none)
_Replay = tuple[Callable, _State | None]


class _Rotation:
    """What ``_known`` returns for a slot that only rotates: stateless, and
    drawing no random bits."""

    rounding = RoundMode.RNE


_ROTATION_ONLY = _Rotation()


def _known(q: object) -> Quant | ScaledQuant | BlockQuant | _Rotation | None:
    """Return the mptorch quantizer that slot ``q`` runs, if it is one known
    to be stateless, and ``None`` otherwise. A rotation (``_RotatedQuant``)
    is stateless, so a rotated slot is known when its quantizer is, and a
    slot that only rotates is ``_ROTATION_ONLY``."""
    if isinstance(q, _RotatedQuant):
        if q.quant is None:
            return _ROTATION_ONLY
        q = q.quant
    if isinstance(q, _CountingQuant):  # simulate(events=True) counting a Quant's casts
        q = q.quant
    if isinstance(q, Quantizer):
        q = q.forward_quant
    if isinstance(q, Quant):
        return None if q.inplace else q
    return q if isinstance(q, ScaledQuant | BlockQuant) else None


def _device_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.mps.get_rng_state()


def _set_device_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.mps.set_rng_state(state)


def _capture(x: torch.Tensor, every: bool) -> _State:
    """Capture the state of the generators a quantizer of ``x`` can draw from.

    An mptorch quantizer (``every=False``) draws from one generator: the
    CUDA device's for a CUDA tensor, the CPU's otherwise. A caller's callable
    (``every=True``) may draw from both the CPU generator and ``x``'s device
    generator, so both are captured."""
    kind = x.device.type
    cpu = torch.get_rng_state() if every or kind != "cuda" else None
    if kind == "cuda" or (kind == "mps" and every):
        return cpu, ((x.device, _device_state(x.device)),)
    return cpu, ()


def quantize_operand(
    x: "torch.Tensor | BlockPacked",
    q: Callable | None,
    flag: bool | None,
    default: bool,
    needed: bool,
) -> tuple[Any, Any, _Replay | None]:
    """Quantize one operand, and decide what backward keeps of it.

    ``q`` is the quantizer slot (``None`` means no quantization), ``flag`` the
    formats' recompute flag, ``default`` what a ``None`` flag means for this
    operand, and ``needed`` whether any gradient reads the operand.

    Returns ``(q_x, saved, replay)``:

    * ``q_x``, the quantized operand, for the forward's math;
    * ``saved``, what to save for backward, or ``None`` when no gradient reads
      the operand, so that the forward saves nothing its backward never reads;
    * ``replay``: ``None`` if ``saved`` is the quantized value, or the
      ``(quantizer, generator states)`` that turn the raw ``saved`` back into
      it.

    An operand that is not a tensor (a frozen layer's ``BlockPacked`` weight)
    is passed through as it is.
    """
    if not q:
        return x, (x if needed else None), None
    if type(x) is BlockPacked:
        # Already quantized: a frozen layer's packed weight (QPackedLinear),
        # whose codes are what its slot produced before the freeze. The test
        # comes after the slot test, so that a layer without quantizers pays
        # nothing for it, and compares the type, which costs about half as
        # much as an isinstance check against torch.Tensor.
        return x, (x if needed else None), None
    # Call _known only where its answer matters: whether a None flag may
    # recompute, and whether a True flag must restore the generators.
    recompute, known = False, None
    if needed and (default if flag is None else flag):
        known = _known(q)
        recompute = flag is not None or known is not None
    state = None
    if recompute and (known is None or known.rounding.is_stochastic):
        if torch.compiler.is_compiling():
            # under compile, keep the copy instead (see the module docstring)
            recompute = False
        else:
            # x is a tensor here: a BlockPacked returned above
            state = _capture(cast(torch.Tensor, x), known is None)
    q_x = q(x)
    if not needed:
        return q_x, None, None
    if recompute and isinstance(q_x, torch.Tensor):
        return q_x, x, (q, state)
    return q_x, q_x, None


def restore(saved: Any, replay: _Replay | None) -> Any:
    """Return the quantized operand from what ``quantize_operand`` saved.

    That is ``saved`` itself, or the quantizer run again on the raw operand,
    with the generators set to the state the forward's call saw and then set
    back to their current state."""
    if replay is None:
        return saved
    q, state = replay
    # the replay repeats casts the forward counted (simulate(events=True),
    # which compiled code never runs)
    counted = not torch.compiler.is_compiling()
    if counted:
        events._replaying += 1
    try:
        if state is None:
            return q(saved)
        cpu, devices = state
        cpu_now = torch.get_rng_state() if cpu is not None else None
        devices_now = [(d, _device_state(d)) for d, _ in devices]
        try:
            if cpu is not None:
                torch.set_rng_state(cpu)
            for d, saved_state in devices:
                _set_device_state(d, saved_state)
            return q(saved)
        finally:
            if cpu_now is not None:
                torch.set_rng_state(cpu_now)
            for d, now in devices_now:
                _set_device_state(d, now)
    finally:
        if counted:
            events._replaying -= 1


def save(ctx, a: object, b: object) -> None:
    """Save the two operands for backward.

    Tensors go through ``save_for_backward``, so that an in-place write to one
    before backward raises an error. Anything else (a ``BlockPacked``), which
    ``save_for_backward`` does not accept, is stored on ``ctx``."""
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    if (ta or a is None) and (tb or b is None):
        ctx.save_for_backward(a, b)
        ctx.saved_structs = None
    else:
        ctx.save_for_backward(a if ta else None, b if tb else None)
        ctx.saved_structs = (None if ta else a, None if tb else b)


def load(ctx) -> tuple[Any, Any]:
    """Return the two operands that ``save`` saved."""
    a, b = ctx.saved_tensors
    structs = ctx.saved_structs
    if structs is None:
        return a, b
    return (a if structs[0] is None else structs[0]), (b if structs[1] is None else structs[1])
