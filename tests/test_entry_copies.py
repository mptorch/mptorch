"""The GEMM-shaped ops' entry points, which are kept as copies, stay copies.

Each GEMM-shaped op has an entry point on every backend (``cpu/``, ``cuda/``,
``mps/``) and a Meta kernel (``meta/``), each in a file named like the CPU
entry file that holds the op, and their bodies are kept identical by hand:

- The CPU, CUDA and MPS bodies are the same text. Each file's ``Backend``
  alias names its backend, which the driver call takes as
  ``run_*<Backend>(``.
- A Meta kernel is the CPU body with that driver call replaced by its
  ``meta_*`` twin from ``common/meta_host.h``.
- An odd op's CPU, CUDA and MPS entry points (``carrier_rounding=RoundMode.RO``)
  forward their arguments to a shared body in ``common/carrier_odd_*host.h``,
  and its Meta kernel is a copy of that body instead.

This test reads the sources, so it needs no build. It compares the copies
token for token, ignoring comments and layout, and lists each mismatch by op
and file. A difference that is deliberate is in ``DELIBERATE_FUNCTIONS`` or
``DELIBERATE_STATEMENTS``, with its reason.
"""

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parents[1] / "mptorch" / "csrc"
BACKENDS = ("cpu", "cuda", "mps", "meta")
ENTRY_FILES = (
    "custom_conv_entry.cpp",
    "custom_conv_odd_entry.cpp",
    "custom_matmul_accumulated_entry.cpp",
    "custom_matmul_block_entry.cpp",
    "custom_matmul_cross_entry.cpp",
    "custom_matmul_entry.cpp",
    "custom_matmul_odd_entry.cpp",
)
ODD_HOST_FILES = ("carrier_odd_host.h", "carrier_odd_gemm_host.h", "carrier_odd_conv_host.h")

# Functions one backend's file has and the CPU's does not: (backend, file) ->
# {op: reason}.
DELIBERATE_FUNCTIONS = {
    ("mps", "custom_matmul_odd_entry.cpp"): {
        "stochastic_cast": "_stochastic_cast is test-only and has no Metal kernel, "
        "so its MPS entry point only raises",
    },
}
# Statements one backend's body has and the CPU's does not, each named by its
# first tokens: (backend, op) -> (first tokens, reason). The statement runs to
# the first `;` outside parentheses.
DELIBERATE_STATEMENTS = {
    ("mps", "block_matmul"): (
        "TORCH_CHECK ( ! binary64 ,",
        "MPS has no float64 tensors, so its entry point refuses a binary64 block "
        "GEMM before the driver runs",
    ),
}

_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|\w+|::|->|&&|\|\||\S', re.S
)


def _tokens(text: str) -> list[str]:
    """C++ tokens, comments dropped; a string literal is one token."""
    return [t for t in _TOKEN.findall(text) if not t.startswith(("//", "/*"))]


def _definitions(text: str, name_pattern: str) -> dict[str, tuple[list[str], list[str]]]:
    """Every top-level function whose name matches ``name_pattern`` (with one
    group, the key), as (parameter tokens, body tokens). A function defined
    inside a namespace counts as top level."""
    toks, out, depth, i = _tokens(text), {}, 0, 0
    while i < len(toks):
        m = re.fullmatch(name_pattern, toks[i])
        if m and depth <= 1 and toks[i + 1] == "(" and toks[i - 1] not in (".", "->", "::"):
            j, paren = i + 1, 0
            while True:
                paren += (toks[j] == "(") - (toks[j] == ")")
                if paren == 0:
                    break
                j += 1
            if toks[j + 1] == "{":
                k, brace = j + 1, 0
                while True:
                    brace += (toks[k] == "{") - (toks[k] == "}")
                    if brace == 0:
                        break
                    k += 1
                out[m.group(1)] = (toks[i + 1 : j + 1], toks[j + 1 : k + 1])
                i = k + 1
                continue
        depth += (toks[i] == "{") - (toks[i] == "}")
        i += 1
    return out


def _normalised(toks: list[str]) -> str:
    """The tokens with each backend's spellings replaced by one: a backend
    type by ``Backend``, a driver call (``run_X<Backend>(``, ``X<Backend>(``
    or the Meta kernel's ``meta_X(``) by ``DRIVER(X)(``."""
    s = " ".join(toks)
    s = s.replace("at :: Tensor", "Tensor")
    s = re.sub(r"\bmptorch :: gemm_(cpu|cuda|mps) :: (Cpu|Cuda|Mps)Backend\b", "Backend", s)
    s = re.sub(r"\bmptorch :: gemm :: (?=run_)", "", s)
    s = re.sub(r"\b(?:run_)?(\w+) < Backend > \(", r"DRIVER(\1) (", s)
    s = re.sub(r"\bmeta_(\w+) \(", r"DRIVER(\1) (", s)
    return s


def _without(body: str, first: str) -> str | None:
    """``body`` without the statement that starts with ``first``, or None if
    there is no such statement."""
    start = body.find(first)
    if start < 0:
        return None
    depth, i = 0, start
    while not (body[i] == ";" and depth == 0):
        depth += (body[i] == "(") - (body[i] == ")")
        i += 1
    return (body[:start] + body[i + 1 :]).replace("  ", " ")


def _first_difference(a: str, b: str) -> str:
    ta, tb = a.split(), b.split()
    k = next(
        (i for i, (x, y) in enumerate(zip(ta, tb, strict=False)) if x != y), min(len(ta), len(tb))
    )
    want, got = (" ".join(t[max(0, k - 5) : k + 6]) for t in (ta, tb))
    return f"at token {k}: expected ...{want}..., found ...{got}..."


def _odd_host_bodies(texts: dict[str, str]) -> dict[str, tuple[list[str], list[str]]]:
    bodies = {}
    for name in ODD_HOST_FILES:
        bodies.update(_definitions(texts[name], r"run_(\w+)"))
    return bodies


def _mismatches(entry: str, texts: dict[str, str]) -> list[str]:
    """Every way the four copies of ``entry`` differ. ``texts`` maps
    ``<backend>/<entry>`` and each odd host header's name to its source."""
    defs = {be: _definitions(texts[f"{be}/{entry}"], rf"(\w+)_{be}") for be in BACKENDS}
    host = _odd_host_bodies(texts)
    problems = []
    for be in BACKENDS[1:]:
        extra = set(defs[be]) - set(defs["cpu"])
        allowed = DELIBERATE_FUNCTIONS.get((be, entry), {})
        problems += [
            f"{be}/{entry} defines {n}, which cpu/{entry} does not"
            for n in sorted(extra - set(allowed))
        ]
        problems += [
            f"{be}/{entry} no longer defines the deliberate {n}"
            for n in sorted(set(allowed) - extra)
        ]
        problems += [
            f"{be}/{entry} lacks {n}, which cpu/{entry} defines"
            for n in sorted(set(defs["cpu"]) - set(defs[be]))
        ]
    for op, (params, body) in defs["cpu"].items():
        cpu = _normalised(params + body)
        forward = re.fullmatch(r"\{ return DRIVER\((\w+_odd)\) \( (.*) \) ; \}", _normalised(body))
        for be in BACKENDS[1:]:
            if op not in defs[be]:
                continue
            got = _normalised(defs[be][op][0] + defs[be][op][1])
            want = cpu
            if be == "meta" and forward:
                # An odd op: the Meta kernel copies the shared body the CPU
                # entry point forwards to.
                shared = forward.group(1)
                if shared not in host:
                    problems.append(
                        f"cpu/{entry}: {op} forwards to run_{shared}, "
                        "which no odd host header defines"
                    )
                    continue
                want = _normalised(host[shared][0] + host[shared][1])
            if (be, op) in DELIBERATE_STATEMENTS:
                first, _ = DELIBERATE_STATEMENTS[(be, op)]
                stripped = _without(got, first)
                if stripped is None:
                    problems.append(
                        f"{be}/{entry}: {op} no longer has the deliberate statement {first!r}"
                    )
                    continue
                got = stripped
            if got != want:
                against = (
                    f"common/carrier_odd_*host.h's run_{forward.group(1)}"
                    if be == "meta" and forward
                    else f"cpu/{entry}"
                )
                problems.append(
                    f"{be}/{entry}: {op} differs from {against} {_first_difference(want, got)}"
                )
        if forward:
            # The forwarder passes its parameters, in order and unchanged.
            names = [p.split()[-1] for p in " ".join(params[1:-1]).split(" , ")]
            if forward.group(2).split(" , ") != names:
                problems.append(f"cpu/{entry}: {op} does not forward its parameters in order")
    return problems


def _sources(entry: str) -> dict[str, str]:
    texts = {f"{be}/{entry}": (CSRC / be / entry).read_text() for be in BACKENDS}
    for name in ODD_HOST_FILES:
        texts[name] = (CSRC / "common" / name).read_text()
    return texts


def test_every_backend_has_the_same_entry_files():
    for be in BACKENDS:
        assert sorted(p.name for p in (CSRC / be).glob("custom_*_entry.cpp")) == list(
            ENTRY_FILES
        ), be


@pytest.mark.parametrize("entry", ENTRY_FILES)
def test_the_copies_are_identical(entry):
    problems = _mismatches(entry, _sources(entry))
    assert not problems, "\n".join(problems)


def test_a_changed_copy_is_reported():
    """Negative control: one swapped argument in one copy is reported, by op
    and file, for a plain copy and for an odd Meta kernel's."""
    entry = "custom_matmul_entry.cpp"
    texts = _sources(entry)
    key = f"cuda/{entry}"
    texts[key] = texts[key].replace("acc_K, acc_P,", "acc_P, acc_K,", 1)
    problems = _mismatches(entry, texts)
    assert len(problems) == 1 and problems[0].startswith(f"cuda/{entry}: binaryK_matmul differs"), (
        problems
    )

    entry = "custom_conv_odd_entry.cpp"
    texts = _sources(entry)
    key = f"meta/{entry}"
    texts[key] = texts[key].replace('"custom_conv_binaryK_odd"', '"custom_conv_superfp_odd"', 1)
    problems = _mismatches(entry, texts)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"meta/{entry}: binaryK_conv_odd differs from common/"), problems


def test_a_deliberate_difference_that_disappears_is_reported():
    entry = "custom_matmul_block_entry.cpp"
    texts = _sources(entry)
    key = f"mps/{entry}"
    texts[key] = re.sub(r"TORCH_CHECK\(!binary64,[^;]*;", "", texts[key], count=1)
    assert _mismatches(entry, texts) == [
        f"mps/{entry}: block_matmul no longer has the deliberate statement "
        "'TORCH_CHECK ( ! binary64 ,'"
    ]
