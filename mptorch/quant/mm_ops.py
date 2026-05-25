"""Unified matrix-multiply ops via torch.ops.mptorch."""

from __future__ import annotations

import warnings

import torch

from mptorch import RoundMode, SubnormalsMode

__all__ = [
    "call_fp_mm",
    "call_superfp_mm",
    "call_fxp_mm",
    "float_mm_v2",
    "float_bmm_v2",
    "superfp_mm_v2",
    "superfp_bmm_v2",
    "fxp_mm_v2",
    "fxp_bmm_v2",
    "has_unified_mm_ops",
]


def has_unified_mm_ops() -> bool:
    return hasattr(torch.ops, "mptorch") and hasattr(torch.ops.mptorch, "fp_mm")


def _round_mode_value(rounding: str) -> int:
    if rounding == "SR":
        return RoundMode.SR.value
    return RoundMode.RNE.value


def _subnormals_mode_value(subnormals: bool) -> int:
    return (
        SubnormalsMode.SUBNORMALS.value
        if subnormals
        else SubnormalsMode.NORMALS.value
    )


def _prepare_bmm_layout(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Mirror v1 float_bmm rank dispatch. Requires a.shape[-1] == b.shape[-2]."""
    ra, rb = a.dim(), b.dim()
    if ra == 3 and rb == 3:
        return a, b, (a.shape[0], a.shape[1], b.shape[2])
    if ra == 3 and rb == 2:
        a_flat = a.reshape(a.shape[0] * a.shape[1], a.shape[2])
        return a_flat, b, (a.shape[0], a.shape[1], b.shape[1])
    if ra == 4 and rb == 4:
        if a.shape[:2] != b.shape[:2]:
            raise ValueError("Wrong tensor sizes for batched MM")
        a_flat = a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
        b_flat = b.reshape(b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
        return a_flat, b_flat, (a.shape[0], a.shape[1], a.shape[2], b.shape[3])
    if ra == 2 and rb == 2:
        return a, b, (a.shape[0], b.shape[1])
    raise ValueError("Wrong tensor sizes for batched MM")


def _default_sr_rbits(
    rounding: str,
    man_add: int,
    man_mul: int,
    rbits_add: int,
    rbits_mul: int,
) -> tuple[int, int]:
    if rounding == "SR":
        if rbits_add <= 0:
            rbits_add = 23 - man_add
        if rbits_mul <= 0:
            rbits_mul = 23 - man_mul
    return rbits_add, rbits_mul


def call_fp_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    man_add: int,
    exp_add: int,
    man_mul: int,
    exp_mul: int,
    rounding: str = "RNE",
    fma: bool = True,
    subnormals: bool = True,
    saturate: bool = True,
    compensated: bool = False,
    rbits_add: int = 0,
    rbits_mul: int = 0,
) -> bool:
    """Call unified fp_mm op. Returns True if dispatched, False for fallback."""
    if not has_unified_mm_ops():
        return False

    man_fma = man_add
    exp_fma = exp_add
    rbits_fma = rbits_add if fma else 0
    torch.ops.mptorch.fp_mm(
        out,
        a.contiguous(),
        b.contiguous(),
        man_add,
        exp_add,
        man_mul,
        exp_mul,
        man_fma,
        exp_fma,
        _round_mode_value(rounding),
        _subnormals_mode_value(subnormals),
        saturate,
        compensated,
        fma,
        rbits_add,
        rbits_mul,
        rbits_fma,
    )
    return True


def call_superfp_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    man_add: int,
    exp_add: int,
    man_mul: int,
    exp_mul: int,
    binades_add_l: int,
    binades_add_u: int,
    binades_mul_l: int,
    binades_mul_u: int,
    man_fma: int,
    exp_fma: int,
    binades_fma_l: int,
    binades_fma_u: int,
    saturate: bool,
    use_fma: bool,
) -> bool:
    if not has_unified_mm_ops():
        return False

    torch.ops.mptorch.superfp_mm(
        out,
        a.contiguous(),
        b.contiguous(),
        man_add,
        exp_add,
        man_mul,
        exp_mul,
        man_fma,
        exp_fma,
        binades_add_l,
        binades_add_u,
        binades_mul_l,
        binades_mul_u,
        binades_fma_l,
        binades_fma_u,
        saturate,
        use_fma,
    )
    return True


def call_fxp_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    wl_add: int,
    fl_add: int,
    wl_mul: int,
    fl_mul: int,
    wl_fma: int,
    fl_fma: int,
    rounding: str,
    symmetric: bool,
    use_fma: bool,
) -> bool:
    if not has_unified_mm_ops():
        return False

    round_mode = RoundMode.SR.value if rounding == "SR" else RoundMode.RNE.value
    torch.ops.mptorch.fxp_mm(
        out,
        a.contiguous(),
        b.contiguous(),
        wl_add,
        fl_add,
        wl_mul,
        fl_mul,
        wl_fma,
        fl_fma,
        round_mode,
        symmetric,
        use_fma,
    )
    return True


def _warn_legacy_mm(name: str) -> None:
    warnings.warn(
        f"{name} via pybind is deprecated; use torch.ops.mptorch unified MM ops",
        DeprecationWarning,
        stacklevel=3,
    )


def float_mm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 23,
    exp_add: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    rounding: str = "RNE",
    fma: bool = True,
    subnormals: bool = True,
    saturate: bool = True,
    compensated: bool = False,
    rbits_add: int = 0,
    rbits_mul: int = 0,
) -> torch.Tensor:
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    if call_fp_mm(
        c,
        a,
        b,
        man_add=man_add,
        exp_add=exp_add,
        man_mul=man_mul,
        exp_mul=exp_mul,
        rounding=rounding,
        fma=fma,
        subnormals=subnormals,
        saturate=saturate,
        compensated=compensated,
        rbits_add=rbits_add,
        rbits_mul=rbits_mul,
    ):
        return c
    raise RuntimeError("float_mm_v2 requires torch.ops.mptorch.fp_mm")


def float_bmm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 23,
    exp_add: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    rounding: str = "RNE",
    fma: bool = True,
    subnormals: bool = True,
    saturate: bool = True,
    compensated: bool = False,
    rbits_add: int = 0,
    rbits_mul: int = 0,
) -> torch.Tensor:
    rbits_add, rbits_mul = _default_sr_rbits(
        rounding, man_add, man_mul, rbits_add, rbits_mul
    )
    a_prep, b_prep, out_shape = _prepare_bmm_layout(a, b)
    c = torch.zeros(out_shape, device=a.device, dtype=a.dtype)
    if call_fp_mm(
        c,
        a_prep,
        b_prep,
        man_add=man_add,
        exp_add=exp_add,
        man_mul=man_mul,
        exp_mul=exp_mul,
        rounding=rounding,
        fma=fma,
        subnormals=subnormals,
        saturate=saturate,
        compensated=compensated,
        rbits_add=rbits_add,
        rbits_mul=rbits_mul,
    ):
        return c
    raise RuntimeError("float_bmm_v2 requires torch.ops.mptorch.fp_mm")


def superfp_mm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int,
    exp_add: int,
    man_mul: int,
    exp_mul: int,
    binades_add_l: int,
    binades_add_u: int,
    binades_mul_l: int,
    binades_mul_u: int,
    man_fma: int,
    exp_fma: int,
    binades_fma_l: int,
    binades_fma_u: int,
    saturate: bool,
    use_fma: bool,
) -> torch.Tensor:
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    if call_superfp_mm(
        c,
        a,
        b,
        man_add=man_add,
        exp_add=exp_add,
        man_mul=man_mul,
        exp_mul=exp_mul,
        binades_add_l=binades_add_l,
        binades_add_u=binades_add_u,
        binades_mul_l=binades_mul_l,
        binades_mul_u=binades_mul_u,
        man_fma=man_fma,
        exp_fma=exp_fma,
        binades_fma_l=binades_fma_l,
        binades_fma_u=binades_fma_u,
        saturate=saturate,
        use_fma=use_fma,
    ):
        return c
    raise RuntimeError("superfp_mm_v2 requires torch.ops.mptorch.superfp_mm")


def superfp_bmm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int,
    exp_add: int,
    man_mul: int,
    exp_mul: int,
    binades_add_l: int,
    binades_add_u: int,
    binades_mul_l: int,
    binades_mul_u: int,
    man_fma: int,
    exp_fma: int,
    binades_fma_l: int,
    binades_fma_u: int,
    saturate: bool,
    use_fma: bool,
) -> torch.Tensor:
    a_prep, b_prep, out_shape = _prepare_bmm_layout(a, b)
    c = torch.zeros(out_shape, device=a.device, dtype=a.dtype)
    if call_superfp_mm(
        c,
        a_prep,
        b_prep,
        man_add=man_add,
        exp_add=exp_add,
        man_mul=man_mul,
        exp_mul=exp_mul,
        binades_add_l=binades_add_l,
        binades_add_u=binades_add_u,
        binades_mul_l=binades_mul_l,
        binades_mul_u=binades_mul_u,
        man_fma=man_fma,
        exp_fma=exp_fma,
        binades_fma_l=binades_fma_l,
        binades_fma_u=binades_fma_u,
        saturate=saturate,
        use_fma=use_fma,
    ):
        return c
    raise RuntimeError("superfp_bmm_v2 requires torch.ops.mptorch.superfp_mm")


def fxp_mm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    wl_add: int,
    fl_add: int,
    wl_mul: int,
    fl_mul: int,
    wl_fma: int,
    fl_fma: int,
    rounding: str,
    symmetric: bool,
    use_fma: bool,
) -> torch.Tensor:
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    if call_fxp_mm(
        c,
        a,
        b,
        wl_add=wl_add,
        fl_add=fl_add,
        wl_mul=wl_mul,
        fl_mul=fl_mul,
        wl_fma=wl_fma,
        fl_fma=fl_fma,
        rounding=rounding,
        symmetric=symmetric,
        use_fma=use_fma,
    ):
        return c
    raise RuntimeError("fxp_mm_v2 requires torch.ops.mptorch.fxp_mm")


def fxp_bmm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    wl_add: int,
    fl_add: int,
    wl_mul: int,
    fl_mul: int,
    wl_fma: int,
    fl_fma: int,
    rounding: str,
    symmetric: bool,
    use_fma: bool,
) -> torch.Tensor:
    a_prep, b_prep, out_shape = _prepare_bmm_layout(a, b)
    c = torch.zeros(out_shape, device=a.device, dtype=a.dtype)
    if call_fxp_mm(
        c,
        a_prep,
        b_prep,
        wl_add=wl_add,
        fl_add=fl_add,
        wl_mul=wl_mul,
        fl_mul=fl_mul,
        wl_fma=wl_fma,
        fl_fma=fl_fma,
        rounding=rounding,
        symmetric=symmetric,
        use_fma=use_fma,
    ):
        return c
    raise RuntimeError("fxp_bmm_v2 requires torch.ops.mptorch.fxp_mm")
