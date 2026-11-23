"""
``simulate`` under ``torch.compile``.

The claims checked here, under ``aot_eager``, which keeps every operator's
result as eager computes it:

- the module :meth:`simulate.wrap` returns compiles in one graph, with one
  mptorch Function per simulated product, and gives eager's outputs and
  gradients, stochastic rounding included;
- a ``simulate`` entered around a compiled model gives the same, under
  ``aot_eager`` and under ``eager``, whose graph calls torch functions again
  at run time;
- per-layer formats compile once: a second call does not recompile;
- entering a ``simulate`` inside compiled code raises, naming the fix;
- under Inductor, mptorch's results are eager's and torch's own arithmetic is
  Inductor's.

``dev/benchmarks/timing/compile_scale.py`` measures the same at 32 layers.
"""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import BinaryK, RoundMode
from mptorch.quant import GemmFormats, Quant, Recipe, SplitMac, simulate
from tests.markers import available_devices
from tests.models import MiniLlama
from tests.test_simulate import _formats, _ids, _run, _same, _site_recipe


def _functions(counter: CompileCounterWithBackend) -> int:
    """How many autograd Functions the compiled graphs call."""
    return sum(
        1
        for graph in counter.graphs
        for node in graph.graph.nodes
        if "autograd_function_apply" in str(node.target)
    )


def _llama(device, layers=2):
    torch.manual_seed(0)
    return MiniLlama(layers=layers).to(device), [_ids(device)]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_wrap_compiles_in_one_graph(device, rounding):
    """Each layer's seven linears and its attention's two sites, and the
    head: nineteen products in one graph, and eager's words."""
    torch._dynamo.reset()
    model, inputs = _llama(device)
    sim = simulate(_site_recipe(rounding), model)
    want = _run(model, inputs, device, sim)
    counter = CompileCounterWithBackend("aot_eager")
    wrapped = sim.wrap()
    compiled = torch.compile(wrapped, backend=counter, fullgraph=True)
    got = _run(model, inputs, device, call=compiled)
    wrapped.close()
    assert counter.frame_count == 1
    assert _functions(counter) == 2 * 9 + 1
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("backend", ["aot_eager", "eager"])
def test_entered_around_a_compiled_model(device, backend):
    """``with sim: torch.compile(model)(x)`` gives eager's words too. Under
    the ``eager`` backend the graph calls torch functions at run time, under
    the mode: none of them is in one of the model's modules then, so the mode
    passes them through, and nothing is simulated twice."""
    torch._dynamo.reset()
    model, inputs = _llama(device)
    sim = simulate(_site_recipe(RoundMode.RNE), model)
    want = _run(model, inputs, device, sim)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    got = _run(model, inputs, device, sim, call=compiled)
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.parametrize("device", available_devices)
def test_per_layer_formats_compile_once(device):
    """Four layers in two alternating formats: one graph, no recompile on a
    second call, and eager's words. The compiled calls come first, before any
    eager one, which is where Dynamo's guards on the simulate's state showed
    (a report filled by the first compiled call made the second recompile):
    a compiled call records nothing in the report, and an eager one does."""
    torch._dynamo.reset()
    model, inputs = _llama(device, layers=4)
    recipe = Recipe(linear=_formats(RoundMode.RNE))
    for i in range(0, 4, 2):
        recipe = recipe.override(f"model.layers.{i}.*", linear=_formats(RoundMode.RNE, p=3))
    sim = simulate(recipe, model)
    counter = CompileCounterWithBackend("aot_eager")
    wrapped = sim.wrap()
    compiled = torch.compile(wrapped, backend=counter, fullgraph=True)
    first = _run(model, inputs, device, call=compiled)
    second = _run(model, inputs, device, call=compiled)
    wrapped.close()
    assert counter.frame_count == 1
    assert sim.report().rows == ()
    want = _run(model, inputs, device, sim)
    assert len(sim.report().simulated) == 4 * 7 + 1
    for k in want:
        assert _same(first[k], want[k]) and _same(second[k], want[k]), k


def test_entering_inside_compiled_code_raises():
    """A ``simulate`` installs hooks on entry, which compiled code cannot do
    correctly, so entering one there raises, naming ``wrap``."""
    torch._dynamo.reset()
    model, inputs = _llama("cpu")
    recipe = Recipe(linear=_formats(RoundMode.RNE))

    def run(ids):
        with simulate(recipe, model):
            return model(ids)

    with pytest.raises(Exception, match="compile sim.wrap"):
        torch.compile(run, backend="aot_eager", fullgraph=True)(*inputs)


def test_inductor():
    """Under Inductor the simulated products are eager's, word for word, and
    the arithmetic around them (norms, softmax, residuals) is Inductor's, so
    the outputs are close."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = MiniLlama(layers=1)
    inputs = [_ids("cpu")]
    fmt = BinaryK(8, 4)
    recipe = Recipe(linear=GemmFormats.uniform(SplitMac(fmt, BinaryK(16, 11)), Quant(fmt)))
    sim = simulate(recipe, model)
    with sim, torch.no_grad():
        want = model(*inputs)
    wrapped = sim.wrap()
    with torch.no_grad():
        got = torch.compile(wrapped, backend="inductor", fullgraph=True)(*inputs)
    wrapped.close()
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
