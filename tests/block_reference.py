"""An independent reference for the block formats (``mptorch.BlockFormat``).

Written from the OCP Microscaling Formats (MX) v1.0 specification, NVIDIA's
description of NVFP4, SuperFP's definition (``dev/benchmarks/superfp_values.py``
enumerates it from its code points) and each element format's value set, and
not from the kernels: nothing here calls a cast, and the decode, the rounding and the bit
layout are each spelled out again, in numpy, from their definitions. The
rounding is a search of the element's sorted value set, where the kernels run
the binaryK cast and a clamp; the scale is the OCP formula on the block's
largest magnitude; the layout is a Python integer the codes are OR-ed into.
A reference that shared the kernels' rules would share their bugs (the
roadmap's lesson on T5 and T8), which is why this one is built from the
definitions and the value sets instead.

float64 is exact for everything MX computes here (every value is a dyadic
rational well inside its range); NVFP4's tensor-scaled arithmetic is what the
format defines in float32, so that part is computed in numpy float32.
"""

from fractions import Fraction

import numpy as np
import torch

from mptorch import BinaryK, BlockFormat, RoundMode, ScaleRounding, SubnormalsMode, SuperFP


class Layout:
    """A minifloat code layout: sign, exponent and mantissa widths, bias, and
    what lies below the normal binades. A ``BinaryK``'s exponent field 0 holds
    its subnormals (or, under EXTENDED_NORMALS, one more binade). A
    ``SuperFP``'s top ``normal_binades`` binades are normals, and below them
    its magnitude codes 1, 2, ... are the powers of two counting up to the
    first normal binade, ``(2^exp_bits - normal_binades) * 2^man_bits`` codes
    with the zero's."""

    def __init__(self, f: "BinaryK | SuperFP"):
        self.signed = f.is_signed
        assert f.bias is not None
        self.bias = f.bias
        if isinstance(f, SuperFP):
            self.e, self.m = f.exp_bits, f.man_bits
            self.sub = None
            lowest_normal_field = (1 << self.e) - f.normal_binades
            self.normal_exp = lowest_normal_field - self.bias
            self.n_super = lowest_normal_field << self.m
        else:
            self.e = f.K - f.P + (0 if f.is_signed else 1)
            self.m = f.P - 1
            self.sub = f.subnormals
            self.normal_exp = None
            self.n_super = 0
        self.bits = int(self.signed) + self.e + self.m

    def pow2_rounding(self, v: float) -> bool:
        """Whether the cast rounds a value just above ``v`` with no mantissa,
        between two powers of two: a superfp's supernormals (the last of which
        meets its first normal binade), and every binade of a format with no
        mantissa bits. There a tie goes to the power whose exponent is even,
        not to an even code (``cast_superfp.h`` rounds those on the carrier's
        exponent field, keeping the odd field, which with an odd bias is an
        even exponent; binaryK's ``round_bitwise_nearest_even`` with no
        mantissa does the same)."""
        if self.m == 0:
            return True
        return self.normal_exp is not None and 0 < v < 2.0**self.normal_exp

    def magnitude(self, code: int) -> float:
        """The value of a magnitude code, from the IEEE-style definition:
        ``2^(E - bias) * 1.M`` for a normal, ``2^(1 - bias) * 0.M`` for
        exponent field 0, which EXTENDED_NORMALS reads as ``2^-bias * 1.M``
        except its 1.0, the zero; and a superfp's supernormal code ``c``, the
        power of two ``n_super - c`` binades below its first normal binade."""
        if code < self.n_super:
            assert self.normal_exp is not None
            return (
                0.0 if code == 0 else float(np.ldexp(1.0, self.normal_exp - (self.n_super - code)))
            )
        E, M = code >> self.m, code & ((1 << self.m) - 1)
        frac = M / (1 << self.m)
        if E == 0:
            if self.sub is SubnormalsMode.EXTENDED_NORMALS:
                return 0.0 if M == 0 else float(np.ldexp(1.0 + frac, -self.bias))
            return float(np.ldexp(frac, 1 - self.bias))
        return float(np.ldexp(1.0 + frac, E - self.bias))

    def is_value(self, code: int) -> bool:
        """Whether a magnitude code holds a value of the format: NORMALS keeps
        no subnormals, so exponent field 0 holds its zero alone."""
        if code < self.n_super:
            return True
        E, M = code >> self.m, code & ((1 << self.m) - 1)
        return not (self.sub is SubnormalsMode.NORMALS and E == 0 and M != 0)


def _specials(fmt: BlockFormat) -> set[int]:
    return {c for c in (fmt.nan_code, fmt.inf_code) if c is not None}


def _grid(lay: Layout, top: float, skip: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A layout's non-negative values up to ``top``, sorted, their magnitude
    codes, and whether a value just above each rounds between powers of two."""
    vals, codes = [], []
    for c in range(1 << (lay.e + lay.m)):
        if c in skip or not lay.is_value(c):
            continue
        v = lay.magnitude(c)
        if v <= top and (v > 0 or c == 0):
            vals.append(v)
            codes.append(c)
    order = np.argsort(vals, kind="stable")
    v, c = np.array(vals)[order], np.array(codes)[order]
    return v, c, np.array([lay.pow2_rounding(x) for x in v])


def elem_grid(fmt: BlockFormat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The element's non-negative values up to ``elem_max``, sorted, their
    magnitude codes, and where they round between powers of two."""
    assert fmt.elem_max is not None
    return _grid(Layout(fmt.elem), fmt.elem_max, _specials(fmt))


def decode_elem(fmt: BlockFormat, codes: np.ndarray) -> np.ndarray:
    """Element codes to float64 values: NaN and infinity by their codes, the
    sign bit, and the sign-only code as +0.0 (the library's one zero)."""
    lay = Layout(fmt.elem)
    mag_bits = lay.e + lay.m
    out = np.empty(codes.shape, dtype=np.float64)
    for idx, c in np.ndenumerate(codes):
        c = int(c)
        mag = c & ((1 << mag_bits) - 1)
        neg = lay.signed and (c >> mag_bits) & 1
        if fmt.inf_code is not None and mag == fmt.inf_code:
            v = float("inf")
        elif mag in _specials(fmt) or (
            fmt.inf_code is not None and fmt.nan_code is not None and mag > fmt.inf_code
        ):
            out[idx] = float("nan")
            continue
        else:
            v = lay.magnitude(mag)
        out[idx] = -v if neg and v != 0 else v
    return out


def scale_value(fmt: BlockFormat, code: int, tensor_scale: float) -> float:
    """A scale code's factor. E8M0 (no mantissa): ``2^(code - bias)``, its
    all-ones code NaN. A scale with a mantissa: its value times the tensor
    scale, one float32 product, its all-ones magnitude code NaN."""
    s = fmt.scale
    assert s is not None
    lay = Layout(s)
    nan = (1 << (lay.e + lay.m)) - 1
    if code & nan == nan:
        return float("nan")
    if isinstance(s, BinaryK) and s.P == 1:
        return float(np.ldexp(1.0, code - lay.bias))
    return float(np.float32(np.float32(lay.magnitude(code & nan)) * np.float32(tensor_scale)))


def _round_magnitude(
    a: np.ndarray,
    grid: np.ndarray,
    codes: np.ndarray,
    pow2: np.ndarray,
    mode: RoundMode,
    up: np.ndarray,
):
    """Magnitudes ``a`` (finite, >= 0) rounded onto ``grid``, which holds 0 and
    ends at the largest value, with anything at or above it taking it (a
    saturating format). ``pow2`` marks the grid values above which the format
    rounds between powers of two. ``up`` says, per element, which way a
    directed mode goes on the magnitude (away from zero or toward it). Returns
    the grid index."""
    top = len(grid) - 1
    lo = np.clip(np.searchsorted(grid, a, side="right") - 1, 0, top)
    hi = np.clip(lo + 1, 0, top)
    exact = grid[lo] == a
    over = a >= grid[top]
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    if mode in (RoundMode.RNE, RoundMode.RNA):
        tie = d_lo == d_hi
        if mode is RoundMode.RNE:
            # ties to the even code, and a tie with zero to zero (with a
            # smallest value whose code is even, NORMALS', parity says nothing);
            # between two powers of two, to the one whose exponent is even
            hi_exp = np.frexp(grid[hi])[1] - 1
            tie_hi = np.where(pow2[lo], hi_exp % 2 == 0, codes[hi] % 2 == 0) & (lo != 0)
        else:
            tie_hi = np.ones_like(tie)
        pick = np.where(d_hi < d_lo, hi, lo)
        pick = np.where(tie, np.where(tie_hi, hi, lo), pick)
    elif mode is RoundMode.RO:
        # the odd one of the two; between zero and the smallest value, the
        # smallest value, whatever its code's parity
        pick = np.where((codes[hi] % 2 == 1) | (lo == 0), hi, lo)
    else:
        pick = np.where(up, hi, lo)
    pick = np.where(exact, lo, pick)
    return np.where(over, top, pick)


def round_elements(fmt: BlockFormat, x: np.ndarray, mode: RoundMode) -> np.ndarray:
    """Values already divided by their scale (float64, exact), rounded to the
    element format in ``mode``, saturating at ``elem_max``. Returns the codes
    (sign included); NaN is the caller's."""
    assert fmt.elem_max is not None
    lay = Layout(fmt.elem)
    grid, codes, pow2 = elem_grid(fmt)
    neg = x < 0
    a = np.where(np.isinf(x), fmt.elem_max, np.abs(x))
    if mode is RoundMode.RU:
        up = ~neg
    elif mode is RoundMode.RD:
        up = neg
    else:
        up = np.zeros_like(neg)
    idx = _round_magnitude(a, grid, codes, pow2, mode, up)
    mag = codes[idx]
    if not lay.signed:
        return np.where(neg, 0, mag)
    sign = 1 << (lay.e + lay.m)
    return np.where(neg & (mag != 0), mag | sign, mag)


def fit_threshold(fmt: BlockFormat) -> float:
    """``elem_max`` plus half the step to the value the element format would
    have next if it went on past ``elem_max``: twice ``elem_max`` where the
    format steps between powers of two, the subnormal step below its normals,
    and one unit in the last place of ``elem_max``'s binade otherwise. The
    largest scaled element that rounding to nearest still takes to
    ``elem_max``, which ``ScaleRounding.SELECTIVE`` compares with."""
    lay = Layout(fmt.elem)
    em = fmt.elem_max
    assert em is not None
    if lay.pow2_rounding(em):
        step = em
    elif lay.sub is not SubnormalsMode.EXTENDED_NORMALS and em < 2.0 ** (1 - lay.bias):
        step = 2.0 ** (1 - lay.bias - lay.m)
    else:
        step = 2.0 ** (np.frexp(em)[1] - 1 - lay.m)
    return em + step / 2


def _log2_floor(q: Fraction) -> int:
    """``floor(log2(q))`` of a positive rational."""
    k = q.numerator.bit_length() - q.denominator.bit_length()
    return k if Fraction(2) ** k <= q else k - 1


def _pow2_exponent(fmt: BlockFormat, amax: float) -> int:
    """The shared exponent a rule gives a nonzero finite ``amax``, before the
    clamp to the scale's codes, from the rule's statement in exact
    rationals: OCP's formula, or ``r = amax / elem_max`` rounded between
    powers of two -- the nearest (a tie to the even exponent), the one at or
    above, or the nearest unless ``amax`` at that scale is above the fit
    threshold, then the next."""
    rule = fmt.scale_rule
    if rule is ScaleRounding.OCP:
        return _log2_floor(Fraction(amax)) - fmt.emax
    assert fmt.elem_max is not None
    r = Fraction(amax) / Fraction(fmt.elem_max)
    k = _log2_floor(r)
    if rule is ScaleRounding.UP:
        return k if r == Fraction(2) ** k else k + 1
    mid = Fraction(3, 2) * Fraction(2) ** k
    X = k if r < mid else k + 1 if r > mid else (k if k % 2 == 0 else k + 1)
    if rule is ScaleRounding.SELECTIVE and Fraction(amax) / Fraction(2) ** X > Fraction(
        fit_threshold(fmt)
    ):
        X += 1
    return X


def tensor_scale_of(x: torch.Tensor, fmt: BlockFormat) -> float:
    """NVFP4's default per-tensor scale: ``amax(|x|) / (elem_max * scale_max)``
    over the finite elements, in float32; 1 for a tensor with none."""
    assert fmt.elem_max is not None
    a = x.detach().abs().float().cpu().numpy()
    a = a[np.isfinite(a)]
    amax = np.float32(a.max()) if a.size else np.float32(0)
    ts = float(amax / np.float32(fmt.elem_max * fmt.scale_max))
    return ts if ts > 0 else 1.0


def tile_scale(fmt: BlockFormat, amax: float, has_nan: bool, ts: float) -> tuple[int, float, str]:
    """A tile's scale code and how an element meets it: ``("mul", 2^-X)`` for
    E8M0, ``("div", d)`` for a mantissa scale, ``("none", 1)`` without one.

    By the format's ``scale_rule``. E8M0: ``X`` from ``_pow2_exponent``
    (OCP's ``floor(log2(amax)) - emax`` by default), clamped to its codes, and
    code 0 for a zero block. Any other scale (E4M3, a superfp): the ratio
    ``amax / (elem_max * S_t)`` rounded to nearest even in the scale format
    (NVFP4's rule), or up, or, for SELECTIVE, to nearest unless the block's
    largest element at that scale is above the fit threshold, then up; at
    most its largest value, at least its smallest positive one, all in
    float32. A NaN in a format with no element NaN code makes the scale the
    scale's NaN."""
    s = fmt.scale
    if s is None:
        return 0, 1.0, "none"
    lay = Layout(s)
    nan_code = (1 << (lay.e + lay.m)) - 1
    if isinstance(s, BinaryK) and s.P == 1:
        lo, hi = -lay.bias, nan_code - 1 - lay.bias
        if amax == 0:
            X = lo
        elif np.isinf(amax):
            X = hi
        else:
            X = int(np.clip(_pow2_exponent(fmt, amax), lo, hi))
        code, how = X + lay.bias, ("mul", float(np.ldexp(1.0, -X)))
    else:
        grid, codes, pow2 = _grid(lay, fmt.scale_max, set())
        ratio = np.float32(amax) / np.float32(np.float32(fmt.elem_max) * np.float32(ts))

        def rounded(up: bool) -> int:
            mode = RoundMode.RU if up else RoundMode.RNE
            i = _round_magnitude(np.array([float(ratio)]), grid, codes, pow2, mode, np.full(1, up))[
                0
            ]
            return 1 if grid[i] == 0 else int(i)

        rule = fmt.scale_rule
        idx = rounded(rule is ScaleRounding.UP)
        if rule is ScaleRounding.SELECTIVE:
            factor = np.float32(np.float32(grid[idx]) * np.float32(ts))
            with np.errstate(over="ignore"):
                if np.float32(amax) / factor > fit_threshold(fmt):
                    idx = rounded(True)
        code = int(codes[idx])
        how = ("div", float(np.float32(np.float32(grid[idx]) * np.float32(ts))))
    if has_nan and fmt.nan_code is None:
        code = nan_code
    return code, how[1], how[0]


def reference(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int,
    mode: RoundMode,
    tensor_scale: float | None = None,
):
    """``x`` quantized to ``fmt`` along ``axis`` in a deterministic ``mode``.

    Returns ``(codes, scales, values)``: the element codes ``[B, R, C]`` and
    scale codes ``[B, R_tiles, n_blocks]`` of the tensor with ``axis`` moved
    last and its leading dimensions folded (numpy integer arrays), and the
    decoded values as a float32 tensor of ``x``'s shape.
    """
    assert mode is not RoundMode.SR
    xs = x.detach().float().cpu().numpy().astype(np.float64)
    axis = axis % xs.ndim
    moved = np.moveaxis(xs, axis, -1)
    mshape = moved.shape
    m3 = moved.reshape(-1, *mshape[-2:]) if moved.ndim >= 2 else moved.reshape(1, 1, -1)
    B, R, C = m3.shape
    bs, br = fmt.block_size, fmt.block_rows
    nb, rt = -(-C // bs), -(-R // br)
    ts = 1.0
    if fmt.has_tensor_scale:
        ts = tensor_scale_of(x, fmt) if tensor_scale is None else float(np.float32(tensor_scale))
    codes = np.zeros((B, R, C), dtype=np.int64)
    scales = np.zeros((B, rt, nb if fmt.scale is not None else 0), dtype=np.int64)
    values = np.zeros((B, R, C), dtype=np.float64)
    for b in range(B):
        for t in range(rt):
            for j in range(nb):
                tile = m3[b, t * br : (t + 1) * br, j * bs : (j + 1) * bs]
                nan = np.isnan(tile)
                finite_abs = np.abs(tile[~nan])
                amax = float(finite_abs.max()) if finite_abs.size else 0.0
                sc, factor, how = tile_scale(fmt, amax, bool(nan.any()), ts)
                if fmt.scale is not None:
                    scales[b, t, j] = sc
                if how == "mul":
                    scaled = tile * factor
                elif how == "div":
                    scaled = (tile.astype(np.float32) / np.float32(factor)).astype(np.float64)
                else:
                    scaled = tile.copy()
                c = round_elements(fmt, np.where(nan, 0.0, scaled), mode)
                if fmt.nan_code is not None:
                    c = np.where(nan, fmt.nan_code, c)
                elif fmt.scale is None:
                    top = round_elements(fmt, np.array([fmt.elem_max]), RoundMode.RZ)[0]
                    c = np.where(nan, top, c)
                codes[b, t * br : (t + 1) * br, j * bs : (j + 1) * bs] = c
                v = decode_elem(fmt, c)
                if fmt.scale is not None:
                    with np.errstate(over="ignore", invalid="ignore"):  # past binary32: inf
                        v = (v * scale_value(fmt, sc, ts)).astype(np.float32).astype(np.float64)
                values[b, t * br : (t + 1) * br, j * bs : (j + 1) * bs] = np.where(v == 0, 0.0, v)
    out = torch.from_numpy(values.astype(np.float32).reshape(mshape))
    out = torch.from_numpy(np.ascontiguousarray(np.moveaxis(out.numpy(), -1, axis)))
    return codes, scales, out


def pack_bytes(codes: np.ndarray, fmt: BlockFormat) -> np.ndarray:
    """Codes ``[B, R, C]`` laid out as OCP's bit stream: per block, element ``i``
    at bits ``[i * bits, (i + 1) * bits)`` of a little-endian integer, the last
    block's tail zero. Returns ``[B, R, row_bytes]`` uint8."""
    B, R, C = codes.shape
    bits, bs = fmt.elem_bits, fmt.block_size
    nb = -(-C // bs)
    block_bytes = bs * bits // 8
    out = np.zeros((B, R, nb * block_bytes), dtype=np.uint8)
    for b in range(B):
        for r in range(R):
            for j in range(nb):
                stream = 0
                for i, c in enumerate(codes[b, r, j * bs : (j + 1) * bs]):
                    stream |= int(c) << (i * bits)
                out[b, r, j * block_bytes : (j + 1) * block_bytes] = list(
                    stream.to_bytes(block_bytes, "little")
                )
    return out
