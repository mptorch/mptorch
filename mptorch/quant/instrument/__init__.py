"""Whole-model tools: recipes, converting a model, simulating it, and counting its arithmetic.

A :class:`~mptorch.quant.Recipe` names the formats of each kind of
computation in a model; :func:`~mptorch.quant.convert_` gives them to a
model's layers, :class:`~mptorch.quant.simulate` runs a model in them without
converting it, and :func:`~mptorch.quant.profile` counts its products and
sums by format (``docs/source/instrument.rst`` is the user's account). The
modules are named for what they do (``conversion``, ``simulation``,
``profiling``), so that the functions they define, which ``mptorch.quant``
exports, do not shadow them. Every public name is importable from
:mod:`mptorch.quant`, and this package re-exports none.
"""

# One module per tool:
#
#   recipe       GemmFormats, Rule, Recipe: the formats of each kind, and each pass's hooks
#   conversion   register, convert_, convert, unconvert_ and their reports
#   simulation   simulate, SimulatedModule and their reports: the state, the mode, the hooks
#   _handlers    the calls a simulation intercepts, by kind, and what it does with each
#   profiling    profile and its records: a simulation that counts MACs by format
#   palettes     set_palette_maps_, palette_maps: a format per output neuron, chosen by a map
#   _counting    the counting copies of the opt-in kinds' formats, for simulate(events=True)
