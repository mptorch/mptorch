"""What the layers' autograd Functions keep for backward (phase V).

``CustomArithLinear``, ``CustomArithConvNd`` and ``CustomArithMatmul`` need
each quantized operand again in backward, one per gradient: the quantized
input (or ``a``) for the weight's (``b``'s) gradient and the quantized weight
for the input's. Each can keep the quantized tensor, a fresh copy the size of
the operand, or keep the raw operand and quantize it again in backward. The
second costs one more quantize pass and saves the copy, and that is a whole
saving wherever the raw operand is alive anyway: a weight always is, as the
layer's parameter, and an input is when the op that made it keeps its own
output for its backward (ReLU, softmax, sigmoid) or when it is the data
batch. Each Function says what a ``None`` flag means for it
(``_RECOMPUTE_DEFAULT``).

A quantizer is run again only if it is known to depend on its input and the
random generator alone -- mptorch's ``Quant`` (not in place), ``BlockQuant``
called as a fake quantizer, and a ``Quantizer`` over one of those -- unless
the caller sets the flag to ``True``, which is their promise that their own
callable does too. Any other callable keeps its result saved: a QAT observer
called twice would count a batch twice, or round differently in backward once
its scale had moved. A quantizer whose result is not a tensor (a
``BlockPacked``) keeps it saved too: it is compact already.

Under stochastic rounding the second call must draw what the first drew, so
the generators it reads are captured just before the forward's call and set
back around the backward's, and left as the backward found them: the
gradient quantizers and the math hooks draw exactly what they drew before.
mptorch's CUDA kernels draw from their tensor's device's generator (every
CUDA op runs on that device, ``csrc/cuda/device_guard.h``), and its CPU and
MPS kernels from the CPU generator (``csrc/cpu/utils.h``, and
``draw_cpu_seed`` on MPS); a caller's callable gets the CPU generator and its
tensor's device's.
"""

from collections.abc import Callable
from typing import Any, cast

import torch

from ..block import BlockPacked
from ..mac import BlockQuant, Quant
from .quantizer import Quantizer

# (the CPU generator's state or None, ((device, its generator's state), ...))
_State = tuple[torch.Tensor | None, tuple[tuple[torch.device, torch.Tensor], ...]]
# what backward needs to quantize a raw operand again
_Replay = tuple[Callable, _State | None]


def _known(q: object) -> Quant | BlockQuant | None:
    """The mptorch quantizer a slot runs, if it is one known to be stateless."""
    if isinstance(q, Quantizer):
        q = q.forward_quant
    if isinstance(q, Quant):
        return None if q.inplace else q
    return q if isinstance(q, BlockQuant) else None


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
    """The states of the generators a quantizer of ``x`` can draw from:
    mptorch's draws from one (``every=False``), a caller's callable from
    the CPU's and its device's."""
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

    Returns ``(q_x, saved, replay)``: the quantized operand for the forward's
    math; what to save for backward (``None`` when no gradient reads it,
    which is P2's rule); and ``None`` if ``saved`` is the quantized value, or
    the ``(quantizer, generator states)`` that turns the raw ``saved`` back
    into it. ``q`` is a slot (``None`` is the identity), ``flag`` the
    formats' recompute flag and ``default`` what ``None`` means here. An
    operand that is not a tensor (a frozen layer's ``BlockPacked`` weight) is
    passed through unquantized.
    """
    if not q:
        return x, (x if needed else None), None
    if type(x) is BlockPacked:
        # Quantized already: a frozen layer's packed weight (QPackedLinear),
        # whose codes are what its slot made before the freeze. Tested here,
        # behind the slot, so that a layer without quantizers pays nothing,
        # and by type, which is half an isinstance against torch.Tensor.
        return x, (x if needed else None), None
    # _known only where it decides something: whether None may recompute,
    # and whether True must rewind.
    recompute, known = False, None
    if needed and (default if flag is None else flag):
        known = _known(q)
        recompute = flag is not None or known is not None
    state = None
    if recompute and (known is None or known.rounding.is_stochastic):
        state = _capture(cast(torch.Tensor, x), known is None)  # not a BlockPacked, above
    q_x = q(x)
    if not needed:
        return q_x, None, None
    if recompute and isinstance(q_x, torch.Tensor):
        return q_x, x, (q, state)
    return q_x, q_x, None


def restore(saved: Any, replay: _Replay | None) -> Any:
    """The quantized operand from what ``quantize_operand`` saved: ``saved``
    itself, or the quantizer run again on the raw operand under the
    generators as the forward's call saw them, which are then set back."""
    if replay is None:
        return saved
    q, state = replay
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


def save(ctx, a: object, b: object) -> None:
    """Save the two operands: tensors through ``save_for_backward``, so an
    in-place write to one before backward raises, and anything else (a
    ``BlockPacked``) on ``ctx``, which ``save_for_backward`` does not take."""
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    if (ta or a is None) and (tb or b is None):
        ctx.save_for_backward(a, b)
        ctx.saved_structs = None
    else:
        ctx.save_for_backward(a if ta else None, b if tb else None)
        ctx.saved_structs = (None if ta else a, None if tb else b)


def load(ctx) -> tuple[Any, Any]:
    """The two operands ``save`` saved."""
    a, b = ctx.saved_tensors
    structs = ctx.saved_structs
    if structs is None:
        return a, b
    return (a if structs[0] is None else structs[0]), (b if structs[1] is None else structs[1])
