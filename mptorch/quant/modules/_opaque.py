"""What keeps mptorch's autograd Functions opaque to torch function modes.

A ``TorchFunctionMode`` that intercepts a model's calls (``simulate``, a
profiler) must see a layer's call and none of the torch calls the layer
makes inside, or it would handle those a second time. Each Function
therefore runs its forward under ``torch._C.DisableTorchFunction()``, and its
backward under :func:`_opaque_backward`.
"""

import contextlib

import torch


def _opaque_backward() -> contextlib.AbstractContextManager:
    """The guard a Function's backward runs under: ``DisableTorchFunction``
    in eager, and nothing while Dynamo traces.

    Under ``torch.compile`` (torch 2.15), Dynamo traced a backward that runs
    an mptorch GEMM inside ``DisableTorchFunction`` (a gradient quantizer
    alone did not do it) and then left torch function modes disabled for the
    rest of the forward it was tracing. A mode entered inside the compiled
    code (``simulate``'s) then saw no call after that layer's, and the graph
    ran those calls in plain PyTorch, with no graph break to say so. The
    guard is not needed there: Dynamo's trace of a backward shows a mode none
    of the backward's calls, and the compiled backward runs aten ops, which a
    mode never sees. ``tests/test_function_opacity.py`` checks that a mode
    inside compiled code routes every layer, with eager's results.
    """
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    return torch._C.DisableTorchFunction()
