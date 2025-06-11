import torch
from mptorch import (
    Number,
    FixedPoint,
    FloatingPoint,
    SuperNormalFloat,
    BlockFloatingPoint,
    Binary8,
)
from torch.utils.cpp_extension import load
import os
from typing import Literal

from .cublas import cublas_acceleration

__all__ = [
    "fixed_point_quantize",
    "block_quantize",
    "float_quantize",
    "binary8_quantize",
    "superfp_quantize",
    "quantizer",
    "mp_mm",
    "mp_bmm",
    "float_mm",
    "float_bmm",
    "fxp_mm",
    "fxp_bmm",
    "superfp_mm",
    "superfp_bmm",
    "mp_softmax_forward",
    "mp_softmax_backward",
    "float_softmax_forward",
    "float_softmax_lse_forward",
    "float_softmax_backward",
    "superfp_softmax_forward",
    "superfp_softmax_lse_forward",
    "superfp_softmax_backward",
    "binary8_softmax_forward",
    "binary8_softmax_lse_forward",
    "binary8_softmax_backward",
    "cublas_mm",
    "cublas_bmm",
    "CUBLASMatrixType",
    "CUBLASComputeType",
    "cublas_acceleration",
    "mp_layernorm_forward",
    "mp_layernorm_backward",
    "float_layernorm_forward",
    "float_layernorm_backward",
    "superfp_layernorm_forward",
    "superfp_layernorm_backward",
    "binary8_layernorm_forward",
    "binary8_layernorm_backward",
]

current_path = os.path.dirname(os.path.realpath(__file__))
quant_cpu = load(
    name="quant_cpu",
    sources=[
        os.path.join(current_path, "quant_cpu/pybind_cpu.cpp"),
        os.path.join(current_path, "quant_cpu/binary8.cpp"),
        os.path.join(current_path, "quant_cpu/quant.cpp"),
        os.path.join(current_path, "quant_cpu/sim_helper.cpp"),
        os.path.join(current_path, "quant_cpu/bit_helper.cpp"),
    ],
    extra_cflags=["-fopenmp"],
)

if torch.cuda.is_available():
    extra_ldflags = []
    if os.name == "nt":
        extra_ldflags.append("cublas.lib")
    quant_cuda = load(
        name="quant_cuda",
        sources=[
            os.path.join(current_path, "quant_cuda/pybind_cuda.cpp"),
            os.path.join(current_path, "quant_cuda/bit_helper.cu"),
            os.path.join(current_path, "quant_cuda/sim_helper.cu"),
            os.path.join(current_path, "quant_cuda/block_kernel.cu"),
            os.path.join(current_path, "quant_cuda/fp_kernel.cu"),
            os.path.join(current_path, "quant_cuda/fxp_kernel.cu"),
            os.path.join(current_path, "quant_cuda/superfp_kernel.cu"),
            os.path.join(current_path, "quant_cuda/quant.cu"),
            os.path.join(current_path, "quant_cuda/binary8_kernel.cu"),
            os.path.join(current_path, "quant_cuda/cublas_helper.cpp"),
        ],
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=["--extended-lambda"],
    )
else:
    quant_cuda = quant_cpu


def assert_wl_fl(wl: int, fl: int, stage: str = ""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def get_module(x: torch.Tensor):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module


if torch.cuda.is_available():
    CUBLASComputeType = quant_cuda.CUBLASComputeType
    CUBLASMatrixType = quant_cuda.CUBLASMatrixType
    CUBLASMatrixType = quant_cuda.CUBLASMatrixType
else:
    CUBLASComputeType, CUBLASMatrixType = None, None


def normalize_binades(binades: int | tuple[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(binades, int):
        binades_l, binades_u = binades, binades
    elif len(binades) == 1:
        binades_l, binades_u = binades[0], binades[0]
    else:
        binades_l, binades_u = binades[0], binades[1]

    return (binades_l, binades_u)


def cublas_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    input_type: CUBLASMatrixType,
    output_type: CUBLASMatrixType,
    compute_type: CUBLASComputeType,
    pedantic: bool,
) -> torch.Tensor:
    """
    Python wrapper for floating-point cuBLAS GEMM (`cublasGemmEx`). This function
    only accepts `binary32` input and output matrices, but allows intermediate
    casting to other datatypes supported by cuBLAS. Please see allowed type combinations
    in the [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/#cublasgemmex).

    Args:
        a: the input of GEMM, with shape (M, K)
        b: the input of GEMM, with shape (K, N)
        input_type: intermediate float format for input matrices (`CUBLASMatrixType`)
        output_type: intermedtiate float format for the output matrix (`CUBLASMatrixType`)
        compute_type: accumulator type used by cuBLAS GEMM (`CUBLASMatrixType`)
        pedantic: whether to hint cuBLAS to use pedantic math or not

    Returns:
        The result of GEMM
    """
    if not torch.cuda.is_available():
        raise NotImplementedError("No CUDA-capable device found. Stopping script.")
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    assert a.is_cuda
    quant_cuda.create_cublas_handle()
    dtype = {
        CUBLASMatrixType.F32: torch.float32,
        CUBLASMatrixType.F16: torch.float16,
        CUBLASMatrixType.BF16: torch.bfloat16,
    }
    a = a.to(dtype[input_type])
    b = b.to(dtype[input_type])
    c = torch.zeros(
        b.shape[1], a.shape[0], device=a.device, dtype=dtype[output_type]  # transposed
    )
    quant_cuda.floating_point_mm_cublas(
        a.t().contiguous(),
        b.t().contiguous(),
        c.contiguous(),
        a.shape[0],
        b.shape[1],
        a.shape[1],
        input_type,
        output_type,
        compute_type,
        pedantic,
    )
    return c.t().to(torch.float32)


def cublas_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    input_type: CUBLASMatrixType,
    output_type: CUBLASMatrixType,
    compute_type: CUBLASComputeType,
    pedantic: bool,
) -> torch.Tensor:
    """
    Python wrapper for floating-point cuBLAS BGEMM (`cublasGemmBatchedEx`). This function
    only accepts `binary32` input and output matrices, but allows intermediate
    casting to other datatypes supported by cuBLAS. Please see the allowed combinations
    in the [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex).
    The last dimension in tensor `a` must match the second to last dimension in tensor `b`.

    Args:
        a: the input to the BGEMM call (2D, 3D and 4D shape possible)
        b: the input to the BGEMM call (2D, 3D and 4D shape possible)
        input_type: intermediate float format for input matrices (`CUBLASMatrixType`)
        output_type: intermedtiate float format for the output matrix (`CUBLASMatrixType`)
        compute_type: accumulator type used by cuBLAS batched GEMM (`CUBLASMatrixType`)
        pedantic: whether to hint cuBLAS to use pedantic math or not

    Returns:
        The result of the batched GEMM
    """
    if not torch.cuda.is_available():
        raise NotImplementedError("No CUDA-capable device found. Stopping script.")
    assert a.shape[-1] == b.shape[-2]
    assert a.device == b.device
    assert a.is_cuda
    quant_cuda.create_cublas_handle()
    dtype = {
        CUBLASMatrixType.F32: torch.float32,
        CUBLASMatrixType.F16: torch.float16,
        CUBLASMatrixType.BF16: torch.bfloat16,
    }
    a = a.to(dtype[input_type])
    b = b.to(dtype[input_type])
    if len(a.shape) == 3 and len(b.shape) == 3:
        c = torch.zeros(
            a.shape[0],
            b.shape[2],
            a.shape[1],  # transposed
            device=a.device,
            dtype=dtype[output_type],
        )
        quant_cuda.floating_point_bmm_cublas(
            a.transpose(-2, -1).contiguous(),
            b.transpose(-2, -1).contiguous(),
            c.contiguous(),
            a.shape[1],
            b.shape[2],
            a.shape[2],
            input_type,
            output_type,
            compute_type,
            pedantic,
        )
        c = c.transpose(-2, -1)
    elif len(a.shape) == 3 and len(b.shape) == 2:
        a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
        c_r = torch.zeros(
            b.shape[1], a_r.shape[0], device=a.device, dtype=dtype[output_type]
        )
        quant_cuda.floating_point_mm_cublas(
            a_r.t().contiguous(),
            b.t().contiguous(),
            c_r.contiguous(),
            a_r.shape[0],
            b.shape[1],
            a_r.shape[1],
            input_type,
            output_type,
            compute_type,
            pedantic,
        )
        c_r = c_r.transpose(-2, -1)
        c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
    elif len(a.shape) == 4 and len(b.shape) == 4:
        a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3]))
        b_r = torch.reshape(b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3]))
        c_r = torch.zeros(
            a_r.shape[0],
            b_r.shape[2],
            a_r.shape[1],
            device=a.device,
            dtype=dtype[output_type],
        )
        quant_cuda.floating_point_bmm_cublas(
            a_r.transpose(-2, -1).contiguous(),
            b_r.transpose(-2, -1).contiguous(),
            c_r.contiguous(),
            a_r.shape[1],
            b_r.shape[2],
            a_r.shape[2],
            input_type,
            output_type,
            compute_type,
            pedantic,
        )
        c_r = c_r.transpose(-2, -1)
        c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
    elif len(a.shape) == 2 and len(b.shape) == 2:
        c = torch.zeros(
            b.shape[1], a.shape[0], device=a.device, dtype=dtype[output_type]
        )
        quant_cuda.floating_point_mm_cublas(
            a.t().contiguous(),
            b.t().contiguous(),
            c.contiguous(),
            a.shape[0],
            b.shape[1],
            a.shape[1],
            input_type,
            output_type,
            compute_type,
            pedantic,
        )
        c = c.t()
    else:
        raise Exception("Wrong tensor sizes")
    return c.to(torch.float32)


def match_mac_format_with_cublas_types(
    man_add: int,
    exp_add: int,
    man_mul: int,
    exp_mul: int,
    rounding: Literal["nearest"],
    fma: bool,
    subnormals: bool,
    saturate: bool,
    fast_mode: Literal["f16", "bf16", "tf32", "f32"] = "f32",
):
    """
    Checks if the current floating-point format configuration for
    matrix multiply operations corresponds to what is available in
    cuBLAS accelerated routines. This allows us to switch to these
    functions, when possible.

    Args:
        man_add: mantissa size for addition operations
        exp_add: exponent size for addition operations
        man_mul: mantissa size for multiply operations
        exp_mul: exponent size for multiply operations
        rounding: rounding mode used in operations (RN or SR)
        fma: if operations should be performed using FMA
        subnormals: are subnormals supported or not
        saturate: are overflow values satturated or not
        fast_mode: whether accelerated accumulations should be used in the cuBLAS calls

    Returns:
        The cuBLAS compute types.
    """
    if not torch.cuda.is_available():
        raise NotImplementedError("No CUDA-capable device found. Stopping script.")

    if man_mul != man_add or exp_mul != exp_add:
        return None

    if (rounding, fma, subnormals, saturate) != ("nearest", True, True, False):
        return None

    mt, ct = CUBLASMatrixType, CUBLASComputeType
    if (man_add, exp_add) == (23, 8):
        if fast_mode == "f16":
            return mt.F32, mt.F32, ct.F32_FAST_F16
        elif fast_mode == "bf16":
            return mt.F32, mt.F32, ct.F32_FAST_BF16
        elif fast_mode == "tf32":
            return mt.F32, mt.F32, ct.F32_FAST_TF32
        else:
            return mt.F32, mt.F32, ct.F32
    elif (man_add, exp_add) == (10, 5):
        return mt.F16, mt.F16, ct.F16
    return None


def translate_overflow_policy(
    module,
    overflow_policy: Literal[
        "saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"
    ],
):
    enum_items = {
        "saturate_infty": module.OverflowPolicy.SATURATE_INFTY,
        "saturate_maxfloat": module.OverflowPolicy.SATURATE_MAXFLOAT,
        "saturate_maxfloat2": module.OverflowPolicy.SATURATE_MAXFLOAT2,
    }
    assert (
        overflow_policy in enum_items.keys()
    ), f"invalid overflow policy, {overflow_policy}"
    return enum_items[overflow_policy]


def mp_softmax_forward(a: torch.Tensor, dim: int, formats) -> torch.Tensor:
    off_cfg = formats.fwd_off
    if type(off_cfg) == FloatingPoint:
        if not formats.use_lse:
            assert (
                formats.fwd_exp.subnormals == off_cfg.subnormals
                and formats.fwd_acc.subnormals == off_cfg.subnormals
            )
            assert (
                formats.fwd_exp.saturate == off_cfg.saturate
                and formats.fwd_acc.saturate == off_cfg.saturate
            )
            return float_softmax_forward(
                a,
                dim,
                formats.fwd_exp.man,
                formats.fwd_exp.exp,
                formats.fwd_off.man,
                formats.fwd_off.exp,
                formats.fwd_acc.man,
                formats.fwd_acc.exp,
                formats.fwd_rnd,
                off_cfg.subnormals,
                off_cfg.saturate,
            )
        else:
            assert formats.fwd_lse.subnormals == off_cfg.subnormals
            assert formats.fwd_lse.saturate == off_cfg.saturate
            return float_softmax_lse_forward(
                a,
                dim,
                formats.fwd_off.man,
                formats.fwd_off.exp,
                formats.fwd_lse.man,
                formats.fwd_lse.exp,
                formats.fwd_rnd,
                off_cfg.subnormals,
                off_cfg.saturate,
            )
    elif type(off_cfg) == SuperNormalFloat:
        if not formats.use_lse:
            assert (
                formats.fwd_exp.saturate == off_cfg.saturate
                and formats.fwd_acc.saturate == off_cfg.saturate
            )
            return superfp_softmax_forward(
                a,
                dim,
                formats.fwd_exp.man,
                formats.fwd_exp.exp,
                formats.fwd_exp.binades,
                formats.fwd_off.man,
                formats.fwd_off.exp,
                formats.fwd_off.binades,
                formats.fwd_acc.man,
                formats.fwd_acc.exp,
                formats.fwd_acc.binades,
                formats.fwd_rnd,
                off_cfg.saturate,
            )
        else:
            assert formats.fwd_lse.saturate == off_cfg.saturate
            return superfp_softmax_lse_forward(
                a,
                dim,
                formats.fwd_off.man,
                formats.fwd_off.exp,
                formats.fwd_off.binades,
                formats.fwd_lse.man,
                formats.fwd_lse.exp,
                formats.fwd_lse.binades,
                formats.fwd_rnd,
                off_cfg.saturate,
            )
    elif type(off_cfg) == Binary8:
        if not formats.use_lse:
            assert (
                formats.fwd_exp.subnormals == off_cfg.subnormals
                and formats.fwd_acc.subnormals == off_cfg.subnormals
            )
            return binary8_softmax_forward(
                a,
                dim,
                formats.fwd_exp.P,
                formats.fwd_exp.overflow_policy,
                formats.fwd_exp.signed,
                formats.fwd_off.P,
                formats.fwd_off.overflow_policy,
                formats.fwd_off.signed,
                formats.fwd_acc.P,
                formats.fwd_acc.overflow_policy,
                formats.fwd_acc.signed,
                formats.fwd_rnd,
                off_cfg.subnormals,
            )
        else:
            assert formats.fwd_lse.subnormals == off_cfg.subnormals
            return binary8_softmax_lse_forward(
                a,
                dim,
                formats.fwd_off.P,
                formats.fwd_off.overflow_policy,
                formats.fwd_off.signed,
                formats.fwd_lse.P,
                formats.fwd_lse.overflow_policy,
                formats.fwd_lse.signed,
                formats.fwd_rnd,
                off_cfg.subnormals,
            )
    raise NotImplementedError("Unsupported number format.")


def mp_softmax_backward(
    input: torch.Tensor, grad_output: torch.Tensor, dim: int, formats
) -> torch.Tensor:
    add_cfg = formats.bwd_add
    if type(add_cfg) == FloatingPoint:
        assert formats.bwd_mul.subnormals == add_cfg.subnormals
        assert formats.bwd_mul.saturate == add_cfg.saturate
        return float_softmax_backward(
            input,
            grad_output,
            dim,
            formats.bwd_add.man,
            formats.bwd_add.exp,
            formats.bwd_mul.man,
            formats.bwd_mul.exp,
            formats.bwd_rnd,
            add_cfg.subnormals,
            add_cfg.saturate,
        )
    elif type(add_cfg) == SuperNormalFloat:
        assert formats.bwd_mul.saturate == add_cfg.saturate
        return superfp_softmax_backward(
            input,
            grad_output,
            dim,
            formats.bwd_add.man,
            formats.bwd_add.exp,
            formats.bwd_add.binades,
            formats.bwd_mul.man,
            formats.bwd_mul.exp,
            formats.bwd_mul.binades,
            formats.bwd_rnd,
            add_cfg.saturate,
        )
    elif type(add_cfg) == Binary8:
        assert formats.bwd_mul.subnormals == add_cfg.subnormals
        return binary8_softmax_backward(
            input,
            grad_output,
            dim,
            formats.bwd_add.P,
            formats.bwd_add.overflow_policy,
            formats.bwd_add.signed,
            formats.bwd_mul.P,
            formats.bwd_mul.overflow_policy,
            formats.bwd_mul.signed,
            formats.bwd_rnd,
            add_cfg.subnormals,
        )
    raise NotImplementedError("Unsupported number format.")


def float_softmax_forward(
    input: torch.Tensor,
    dim: int,
    man_exp: int = 23,
    exp_exp: int = 8,
    man_off: int = 23,
    exp_off: int = 8,
    man_acc: int = 23,
    exp_acc: int = 8,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    subnormals: bool = True,
    saturate: bool = False,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.float_quantize_nearest_softmax_forward(
        input.contiguous(),
        output,
        dim,
        man_exp,
        exp_exp,
        man_off,
        exp_off,
        man_acc,
        exp_acc,
        subnormals,
        saturate,
    )
    return output


def float_softmax_lse_forward(
    input: torch.Tensor,
    dim: int,
    man_off: int = 23,
    exp_off: int = 8,
    man_lse: int = 23,
    exp_lse: int = 8,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    subnormals: bool = True,
    saturate: bool = False,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.float_quantize_nearest_softmax_lse_forward(
        input.contiguous(),
        output,
        dim,
        man_off,
        exp_off,
        man_lse,
        exp_lse,
        subnormals,
        saturate,
    )
    return output


def float_softmax_backward(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    dim: int,
    man_add: int = 23,
    exp_add: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    subnormals: bool = True,
    saturate: bool = False,
) -> torch.Tensor:
    assert input.device == grad_output.device
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    grad_input = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.float_quantize_nearest_softmax_backward(
        input.contiguous(),
        grad_output.contiguous(),
        grad_input,
        dim,
        man_add,
        exp_add,
        man_mul,
        exp_mul,
        subnormals,
        saturate,
    )
    return grad_input


def superfp_softmax_forward(
    input: torch.Tensor,
    dim: int,
    man_exp: int = 23,
    exp_exp: int = 8,
    binades_exp: int | tuple[int] | tuple[int, int] = 1,
    man_off: int = 23,
    exp_off: int = 8,
    binades_off: int | tuple[int] | tuple[int, int] = 1,
    man_acc: int = 23,
    exp_acc: int = 8,
    binades_acc: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    saturate: bool = False,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)

    binades_exp_l, binades_exp_u = normalize_binades(binades_exp)
    binades_off_l, binades_off_u = normalize_binades(binades_off)
    binades_acc_l, binades_acc_u = normalize_binades(binades_acc)

    quant_module.superfp_quantize_nearest_softmax_forward(
        input.contiguous(),
        output,
        dim,
        man_exp,
        exp_exp,
        binades_exp_l,
        binades_exp_u,
        man_off,
        exp_off,
        binades_off_l,
        binades_off_u,
        man_acc,
        exp_acc,
        binades_acc_l,
        binades_acc_u,
        saturate,
    )
    return output


def superfp_softmax_lse_forward(
    input: torch.Tensor,
    dim: int,
    man_off: int = 23,
    exp_off: int = 8,
    binades_off: int | tuple[int] | tuple[int, int] = 1,
    man_lse: int = 23,
    exp_lse: int = 8,
    binades_lse: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stohastic"] = "nearest",
    saturate: bool = False,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)

    binades_off_l, binades_off_u = normalize_binades(binades_off)
    binades_lse_l, binades_lse_u = normalize_binades(binades_lse)

    quant_module.superfp_quantize_nearest_softmax_lse_forward(
        input.contiguous(),
        output,
        dim,
        man_off,
        exp_off,
        binades_off_l,
        binades_off_u,
        man_lse,
        exp_lse,
        binades_lse_l,
        binades_lse_u,
        saturate,
    )
    return output


def superfp_softmax_backward(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    dim: int,
    man_add: int = 23,
    exp_add: int = 8,
    binades_add: int | tuple[int] | tuple[int, int] = 1,
    man_mul: int = 23,
    exp_mul: int = 8,
    binades_mul: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    saturate: bool = False,
) -> torch.Tensor:
    assert input.device == grad_output.device
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    grad_input = torch.zeros_like(input)
    quant_module = get_module(input)

    binades_add_l, binades_add_u = normalize_binades(binades_add)
    binades_mul_l, binades_mul_u = normalize_binades(binades_mul)

    quant_module.superfp_quantize_nearest_softmax_backward(
        input.contiguous(),
        grad_output.contiguous(),
        grad_input,
        dim,
        man_add,
        exp_add,
        binades_add_l,
        binades_add_u,
        man_mul,
        exp_mul,
        binades_mul_l,
        binades_mul_u,
        saturate,
    )
    return grad_input


def binary8_softmax_forward(
    input: torch.Tensor,
    dim: int,
    P_exp: int,
    op_exp: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_exp: bool,
    P_off: int,
    op_off: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_off: bool,
    P_acc: int,
    op_acc: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_acc: bool,
    rounding: Literal["nearest"],
    subnormals: bool,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.binary8_quantize_nearest_softmax_forward(
        input.contiguous(),
        output,
        dim,
        P_exp,
        translate_overflow_policy(quant_module, op_exp),
        signed_exp,
        P_off,
        translate_overflow_policy(quant_module, op_off),
        signed_off,
        P_acc,
        translate_overflow_policy(quant_module, op_acc),
        signed_acc,
        subnormals,
    )
    return output


def binary8_softmax_lse_forward(
    input: torch.Tensor,
    dim: int,
    P_off: int,
    op_off: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_off: bool,
    P_lse: int,
    op_lse: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_lse: bool,
    rounding: Literal["nearest"],
    subnormals: bool,
) -> torch.Tensor:
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    output = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.binary8_quantize_nearest_softmax_lse_forward(
        input.contiguous(),
        output,
        dim,
        P_off,
        translate_overflow_policy(quant_module, op_off),
        signed_off,
        P_lse,
        translate_overflow_policy(quant_module, op_lse),
        signed_lse,
        subnormals,
    )
    return output


def binary8_softmax_backward(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    dim: int,
    P_add: int,
    op_add: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_add: bool,
    P_mul: int,
    op_mul: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_mul: bool,
    rounding: Literal["nearest"],
    subnormals: bool,
) -> torch.Tensor:
    assert input.device == grad_output.device
    assert rounding == "nearest", "Only nearest rounding softmax is implemented."
    grad_input = torch.zeros_like(input)
    quant_module = get_module(input)
    quant_module.binary8_quantize_nearest_softmax_backward(
        input.contiguous(),
        grad_output.contiguous(),
        grad_input,
        dim,
        P_add,
        translate_overflow_policy(quant_module, op_add),
        signed_add,
        P_mul,
        translate_overflow_policy(quant_module, op_mul),
        signed_mul,
        subnormals,
    )
    return grad_input


def mp_layernorm_forward(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dims: list[int],
    formats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_cfg = formats.fwd_acc
    if type(acc_cfg) == FloatingPoint:
        return float_layernorm_forward(
            inp,
            weight,
            bias,
            eps,
            dims,
            formats.fwd_acc.man,
            formats.fwd_acc.exp,
            formats.fwd_mul.man,
            formats.fwd_mul.exp,
            formats.fwd_div.man,
            formats.fwd_div.exp,
            formats.fwd_sqrt.man,
            formats.fwd_sqrt.exp,
            formats.fwd_rnd,
            acc_cfg.subnormals,
            acc_cfg.saturate,
        )

    elif type(acc_cfg) == SuperNormalFloat:
        return superfp_layernorm_forward(
            inp,
            weight,
            bias,
            eps,
            dims,
            formats.fwd_acc.man,
            formats.fwd_acc.exp,
            formats.fwd_acc.binades,
            formats.fwd_mul.man,
            formats.fwd_mul.exp,
            formats.fwd_mul.binades,
            formats.fwd_div.man,
            formats.fwd_div.exp,
            formats.fwd_div.binades,
            formats.fwd_sqrt.man,
            formats.fwd_sqrt.exp,
            formats.fwd_sqrt.binades,
            formats.fwd_rnd,
            acc_cfg.saturate,
        )

    elif type(acc_cfg) == Binary8:
        return binary8_layernorm_forward(
            inp,
            weight,
            bias,
            eps,
            dims,
            formats.fwd_acc.P,
            formats.fwd_acc.overflow_policy,
            formats.fwd_acc.signed,
            formats.fwd_mul.P,
            formats.fwd_mul.overflow_policy,
            formats.fwd_mul.signed,
            formats.fwd_div.P,
            formats.fwd_div.overflow_policy,
            formats.fwd_div.signed,
            formats.fwd_sqrt.P,
            formats.fwd_sqrt.overflow_policy,
            formats.fwd_sqrt.signed,
            formats.fwd_rnd,
            acc_cfg.subnormals,
        )

    raise NotImplementedError("Unsupported float type.")


def mp_layernorm_backward(
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dims: list[int],
    formats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_cfg = formats.bwd_acc
    if type(acc_cfg) == FloatingPoint:
        return float_layernorm_backward(
            inp,
            grad_output,
            weight,
            bias,
            mean,
            rstd,
            dims,
            formats.bwd_acc.man,
            formats.bwd_acc.exp,
            formats.bwd_mul.man,
            formats.bwd_mul.exp,
            formats.bwd_div.man,
            formats.bwd_div.exp,
            formats.bwd_rnd,
            acc_cfg.subnormals,
            acc_cfg.saturate,
        )
    elif type(acc_cfg) == SuperNormalFloat:
        return superfp_layernorm_backward(
            inp,
            grad_output,
            weight,
            bias,
            mean,
            rstd,
            dims,
            formats.bwd_acc.man,
            formats.bwd_acc.exp,
            formats.bwd_acc.binades,
            formats.bwd_mul.man,
            formats.bwd_mul.exp,
            formats.bwd_mul.binades,
            formats.bwd_div.man,
            formats.bwd_div.exp,
            formats.bwd_div.binades,
            formats.bwd_rnd,
            acc_cfg.saturate,
        )
    elif type(acc_cfg) == Binary8:
        return binary8_layernorm_backward(
            inp,
            grad_output,
            weight,
            bias,
            mean,
            rstd,
            dims,
            formats.bwd_acc.P,
            formats.bwd_acc.overflow_policy,
            formats.bwd_acc.signed,
            formats.bwd_mul.P,
            formats.bwd_mul.overflow_policy,
            formats.bwd_mul.signed,
            formats.bwd_div.P,
            formats.bwd_div.overflow_policy,
            formats.bwd_div.signed,
            formats.bwd_rnd,
            acc_cfg.subnormals,
        )

    raise NotImplementedError("Unsupported float type.")


def float_layernorm_forward(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dims: list[int],
    man_acc: int = 23,
    exp_acc: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    man_div: int = 23,
    exp_div: int = 8,
    man_sqrt: int = 23,
    exp_sqrt: int = 8,
    rounding: Literal["nearest"] = "nearest",
    subnormals: bool = True,
    saturate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest roudning layernorm is implemented."

    quant_module = get_module(inp)

    reduced_dim = list(range(inp.dim() - len(dims)))
    reduced_shape = [inp.shape[i] for i in reduced_dim]

    mean = torch.zeros(reduced_shape, device=inp.device)
    rstd = torch.zeros(reduced_shape, device=inp.device)
    output = torch.zeros_like(inp, device=inp.device)

    quant_module.float_quantize_layernorm_forward(
        inp.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        output,
        mean,
        rstd,
        eps,
        dims,
        man_acc,
        exp_acc,
        man_mul,
        exp_mul,
        man_div,
        exp_div,
        man_sqrt,
        exp_sqrt,
        subnormals,
        saturate,
    )
    return output, mean, rstd


def superfp_layernorm_forward(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dims: list[int],
    man_acc: int = 23,
    exp_acc: int = 8,
    binades_acc: int | tuple[int] | tuple[int, int] = 1,
    man_mul: int = 23,
    exp_mul: int = 8,
    binades_mul: int | tuple[int] | tuple[int, int] = 1,
    man_div: int = 23,
    exp_div: int = 8,
    binades_div: int | tuple[int] | tuple[int, int] = 1,
    man_sqrt: int = 23,
    exp_sqrt: int = 8,
    binades_sqrt: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    saturate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest roudning layernorm is implemented."

    quant_module = get_module(inp)

    reduced_dim = list(range(inp.dim() - len(dims)))
    reduced_shape = [inp.shape[i] for i in reduced_dim]

    mean = torch.zeros(reduced_shape, device=inp.device)
    rstd = torch.zeros(reduced_shape, device=inp.device)
    output = torch.zeros_like(inp, device=inp.device)

    binades_acc_l, binades_acc_h = normalize_binades(binades_acc)
    binades_mul_l, binades_mul_h = normalize_binades(binades_mul)
    binades_div_l, binades_div_h = normalize_binades(binades_div)
    binades_sqrt_l, binades_sqrt_h = normalize_binades(binades_sqrt)

    quant_module.superfp_quantize_layernorm_forward(
        inp.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        output,
        mean,
        rstd,
        eps,
        dims,
        man_acc,
        exp_acc,
        binades_acc_l,
        binades_acc_h,
        man_mul,
        exp_mul,
        binades_mul_l,
        binades_mul_h,
        man_div,
        exp_div,
        binades_div_l,
        binades_div_h,
        man_sqrt,
        exp_sqrt,
        binades_sqrt_l,
        binades_sqrt_h,
        saturate,
    )
    return output, mean, rstd


def binary8_layernorm_forward(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dims: list[int],
    P_acc: int,
    op_acc: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_acc: bool,
    P_mul: int,
    op_mul: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_mul: bool,
    P_div: int,
    op_div: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_div: bool,
    P_sqrt: int,
    op_sqrt: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_sqrt: bool,
    rounding: Literal["nearest"],
    subnormals: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest rounding layernorm is implemented."

    quant_module = get_module(inp)

    reduced_dim = list(range(inp.dim() - len(dims)))
    reduced_shape = [inp.shape[i] for i in reduced_dim]

    mean = torch.zeros(reduced_shape, device=inp.device)
    rstd = torch.zeros(reduced_shape, device=inp.device)
    output = torch.zeros_like(inp, device=inp.device)

    quant_module.binary8_quantize_layernorm_forward(
        inp.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        output,
        mean,
        rstd,
        eps,
        dims,
        P_acc,
        op_acc,
        signed_acc,
        P_mul,
        op_mul,
        signed_mul,
        P_div,
        op_div,
        signed_div,
        P_sqrt,
        op_sqrt,
        signed_sqrt,
        subnormals,
    )
    return output, mean, rstd


def float_layernorm_backward(
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dims: list[int],
    man_acc: int = 23,
    exp_acc: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    man_div: int = 23,
    exp_div: int = 8,
    rounding: Literal["nearest"] = "nearest",
    subnormals: bool = True,
    saturate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest rounding layernorm is implemented."

    assert inp.device == grad_output.device

    quant_module = get_module(inp)

    grad_input = torch.zeros_like(inp)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)

    quant_module.float_quantize_layernorm_backward(
        inp.contiguous(),
        grad_output.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        mean.contiguous(),
        rstd.contiguous(),
        grad_input,
        grad_weight,
        grad_bias,
        dims,
        man_acc,
        exp_acc,
        man_mul,
        exp_mul,
        man_div,
        exp_div,
        subnormals,
        saturate,
    )
    return grad_input, grad_weight, grad_bias


def superfp_layernorm_backward(
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dims: list[int],
    man_acc: int = 23,
    exp_acc: int = 8,
    binades_acc: int | tuple[int] | tuple[int, int] = 1,
    man_mul: int = 23,
    exp_mul: int = 8,
    binades_mul: int | tuple[int] | tuple[int, int] = 1,
    man_div: int = 23,
    exp_div: int = 8,
    binades_div: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest"] = "nearest",
    saturate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest rounding layernorm is implemented."

    assert inp.device == grad_output.device

    quant_module = get_module(inp)

    grad_input = torch.zeros_like(inp)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)

    binades_acc_l, binades_acc_h = normalize_binades(binades_acc)
    binades_mul_l, binades_mul_h = normalize_binades(binades_mul)
    binades_div_l, binades_div_h = normalize_binades(binades_div)

    quant_module.superfp_quantize_layernorm_backward(
        inp.contiguous(),
        grad_output.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        mean.contiguous(),
        rstd.contiguous(),
        grad_input,
        grad_weight,
        grad_bias,
        dims,
        man_acc,
        exp_acc,
        binades_acc_l,
        binades_acc_h,
        man_mul,
        exp_mul,
        binades_mul_l,
        binades_mul_h,
        man_div,
        exp_div,
        binades_div_l,
        binades_div_h,
        saturate,
    )
    return grad_input, grad_weight, grad_bias


def binary8_layernorm_backward(
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dims: list[int],
    P_acc: int,
    op_acc: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_acc: bool,
    P_mul: int,
    op_mul: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_mul: bool,
    P_div: int,
    op_div: Literal["saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"],
    signed_div: bool,
    rounding: Literal["nearest", "stochastic"],
    subnormals: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert rounding == "nearest", "Only nearest roudning layernorm is implemented."

    assert inp.device == grad_output.device

    quant_module = get_module(inp)

    grad_input = torch.zeros_like(inp)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)

    quant_module.binary8_quantize_layernorm_backward(
        inp.contiguous(),
        grad_output.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        mean.contiguous(),
        rstd.contiguous(),
        grad_input,
        grad_weight,
        grad_bias,
        dims,
        P_acc,
        op_acc,
        signed_acc,
        P_mul,
        op_mul,
        signed_mul,
        P_div,
        op_div,
        signed_div,
        subnormals,
    )
    return grad_input, grad_weight, grad_bias


def mp_mm(
    a: torch.Tensor, b: torch.Tensor, formats, use_forward: bool = True
) -> torch.Tensor:

    if use_forward:  # FWD format configuration
        if formats.fwd_use_default_prec:
            add_cfg, mul_cfg, fma, rnd = (
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                True,
                "nearest",
            )
        else:
            add_cfg, mul_cfg, fma, rnd = (
                formats.fwd_add,
                formats.fwd_mul,
                formats.fwd_fma,
                formats.fwd_rnd,
            )
    else:  # BWD format configuration
        if formats.bwd_use_default_prec:
            add_cfg, mul_cfg, fma, rnd = (
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                True,
                "nearest",
            )
        else:
            add_cfg, mul_cfg, fma, rnd = (
                formats.bwd_add,
                formats.bwd_mul,
                formats.bwd_fma,
                formats.bwd_rnd,
            )

    assert type(add_cfg) == type(
        mul_cfg
    ), "Only the same types are currently support for add and mul operations"
    if type(add_cfg) == FloatingPoint:
        return float_mm(
            a,
            b,
            man_add=add_cfg.man,
            exp_add=add_cfg.exp,
            man_mul=mul_cfg.man,
            exp_mul=mul_cfg.exp,
            rounding=rnd,
            fma=fma,
            subnormals=add_cfg.subnormals,
            saturate=add_cfg.saturate,
            compensated=formats.compensated,
            rbits_add=formats.rbits_add,
            rbits_mul=formats.rbits_mul,
        )
    elif type(add_cfg) == SuperNormalFloat:
        return superfp_mm(
            a,
            b,
            man_add=add_cfg.man,
            exp_add=add_cfg.exp,
            binades_add=add_cfg.binades,
            man_mul=mul_cfg.man,
            exp_mul=mul_cfg.exp,
            binades_mul=mul_cfg.binades,
            rounding=rnd,
            fma=fma,
            saturate=add_cfg.saturate,
        )
    else:  # fixed-point
        return fxp_mm(
            a,
            b,
            wl_add=add_cfg.wl,
            fl_add=add_cfg.fl,
            wl_mul=mul_cfg.wl,
            fl_mul=mul_cfg.fl,
            symmetric=False,
            rounding=rnd,
            fma=fma,
        )


def fxp_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    wl_add: int = 16,
    fl_add: int = 8,
    wl_mul: int = 16,
    fl_mul: int = 8,
    symmetric: bool = False,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
) -> torch.Tensor:
    """
    Mixed-precision fixed-point GEMM with customized formats for the multipliers and accumulators
    Args:
        `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        `wl_add` (int) : word length of the fixed point number being simulated for addition
        `fl_add` (int) : fractional length of the fixed point number being simulated for addition
        `wl_mul` (int) : word length of the fixed point number being simulated for multiplication
        `fl_mul` (int) : fractional length of the fixed point number being simulated for multiplicaiton
        `symmetric` (bool): use a symmetric fixed point encoding
        `rounding` (str): use rounding mode used in the multiply-add operations
        `fma` (bool) : use fma operation instead of separate multiply and add (uses the
        wl_add and fl_add parameters for the rounding of the fma results)
    Returns:
        - the result of GEMM (torch.Tensor)
    """

    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    quant_module = get_module(a)
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
    if rounding == "nearest":
        if not fma:
            quant_module.fixed_point_quantize_nearest_mm(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                wl_add,
                fl_add,
                wl_mul,
                fl_mul,
                symmetric,
            )
        else:
            quant_module.fixed_point_quantize_nearest_mm_fma(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                wl_add,
                fl_add,
                symmetric,
            )
    else:
        if not fma:
            quant_module.fixed_point_quantize_stochastic_mm(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                wl_add,
                fl_add,
                wl_mul,
                fl_mul,
                symmetric,
            )
        else:
            quant_module.fixed_point_quantize_stochastic_mm_fma(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                wl_add,
                fl_add,
                symmetric,
            )
    return c


def float_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 23,
    exp_add: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
    subnormals: bool = True,
    saturate: bool = True,
    compensated: bool = False,
    rbits_add: int = 0,
    rbits_mul: int = 0,
) -> torch.Tensor:
    """
    Mixed-precision floating-point GEMM with customized formats for the multipliers and accumulators
    Args:
        `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        `exp_add` (int) : number of bits allocated for exponent in addition result
        `man_add` (int) : number of bits allocated for mantissa in addition result, not counting the virtual bit
        `exp_mul` (int) : number of bits allocated for exponent in multiplication result
        `man_mul` (int) : number of bits allocated for mantissa in multiplication result, not counting the virtual bit
        `fma` (bool) : use fma operation instead of separate multiply and add (uses the
        man_add and exp_add parameters for the rounding of the fma results)
        `subnormals` (bool): allow the use of subnormal values
        `saturate` (bool): saturate results (i.e., clamp values at min/max representable in the format instead of outputting infinities)
        `compensated` (bool): compensated flag (i.e., use Kahan summation)
        `rbits_add` (int): number of random bits to use in stochastic rounding addition (in case rounding mode is stochastic)
        `rbits_mul` (int): number of random bits to use in stochastic rounding multiplication (in case rounding mode is stochastic)
    Returns:
        - the result of GEMM (torch.Tensor)
    """

    if cublas_acceleration.enabled and a.is_cuda and b.is_cuda:
        types = match_mac_format_with_cublas_types(
            man_add,
            exp_add,
            man_mul,
            exp_mul,
            rounding,
            fma,
            subnormals,
            saturate,
            cublas_acceleration.fast_mode,
        )
        if types is not None:
            input_type, output_type, compute_type = types
            return cublas_mm(a, b, input_type, output_type, compute_type, False)

    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    quant_module = get_module(a)
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
    if rounding == "nearest":
        if not fma:
            quant_module.float_quantize_nearest_mm(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                man_mul,
                exp_mul,
                subnormals,
                saturate,
                compensated,
            )
        else:
            quant_module.float_quantize_nearest_mm_fma(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                subnormals,
                saturate,
                compensated,
            )
    else:
        if rbits_add <= 0:
            rbits_add = 23 - man_add
        if rbits_mul <= 0:
            rbits_mul == 23 - man_mul
        if not fma:
            quant_module.float_quantize_stochastic_mm(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                rbits_add,
                man_mul,
                exp_mul,
                rbits_mul,
                subnormals,
                saturate,
            )
        else:
            quant_module.float_quantize_stochastic_mm_fma(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                rbits_add,
                subnormals,
                saturate,
            )
    return c


def superfp_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 7,
    exp_add: int = 8,
    binades_add: int | tuple[int] | tuple[int, int] = 1,
    man_mul: int = 7,
    exp_mul: int = 8,
    binades_mul: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
    saturate: bool = False,
) -> torch.Tensor:
    """
    Mixed-precision super normal floating-point GEMM with customized formats for the multipliers and accumulators
    Args:
        `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        `exp_add` (int) : number of bits allocated for exponent in addition result
        `man_add` (int) : number of bits allocated for mantissa in addition result, not counting the virtual bit
        `binades_add` (int | tuple[int] | tuple[int, int]) : number of binades in the sub/sup normal range for addition operations
        `exp_mul` (int) : number of bits allocated for exponent in multiplication result
        `man_mul` (int) : number of bits allocated for mantissa in multiplication result, not counting the virtual bit
        `binades_mul` (int | tuple[int] | tuple[int, int]) : number of binades in the sub/sup normal range for multiplication operations
        `fma` (bool) : use fma operation instead of separate multiply and add (uses the
        man_add and exp_add parameters for the rounding of the fma results)
        `saturate` (bool): saturate results (i.e., clamp values at min/max representable
        in the format instead of outputting infinities)
    Returns:
        - the result of GEMM (torch.Tensor)
    """

    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    quant_module = get_module(a)
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device)

    binades_add_l, binades_add_h = normalize_binades(binades_add)
    binades_mul_l, binades_mul_h = normalize_binades(binades_mul)

    if rounding == "nearest":
        if not fma:
            quant_module.superfp_quantize_nearest_mm(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                man_mul,
                exp_mul,
                binades_add_l,
                binades_add_h,
                binades_mul_l,
                binades_mul_h,
                saturate,
            )
        else:
            quant_module.superfp_quantize_nearest_mm_fma(
                a.contiguous(),
                b.contiguous(),
                c.contiguous(),
                a.shape[0],
                b.shape[1],
                a.shape[1],
                man_add,
                exp_add,
                binades_add_l,
                binades_add_h,
                saturate,
            )
    else:
        # TODO: stochastic rounding is not implemented yet
        raise NotImplementedError("SR SuperNormalFloat MM is not yet implemented")
    return c


def mp_bmm(
    a: torch.Tensor, b: torch.Tensor, formats, use_forward: bool = True
) -> torch.Tensor:
    if use_forward:  # FWD format configuration
        if formats.fwd_use_default_prec:
            add_cfg, mul_cfg, fma, rnd = (
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                True,
                "nearest",
            )
        else:
            add_cfg, mul_cfg, fma, rnd = (
                formats.fwd_add,
                formats.fwd_mul,
                formats.fwd_fma,
                formats.fwd_rnd,
            )
    else:  # BWD format configuration
        if formats.bwd_use_default_prec:
            add_cfg, mul_cfg, fma, rnd = (
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                FloatingPoint(exp=8, man=23, subnormals=True, saturate=False),
                True,
                "nearest",
            )
        else:
            add_cfg, mul_cfg, fma, rnd = (
                formats.bwd_add,
                formats.bwd_mul,
                formats.bwd_fma,
                formats.bwd_rnd,
            )

    assert type(add_cfg) == type(
        mul_cfg
    ), "Only the same types are currently support for add and mul operations"

    if type(add_cfg) == FloatingPoint:
        return float_bmm(
            a,
            b,
            man_add=add_cfg.man,
            exp_add=add_cfg.exp,
            man_mul=mul_cfg.man,
            exp_mul=mul_cfg.exp,
            rounding=rnd,
            fma=fma,
            subnormals=add_cfg.subnormals,
            saturate=add_cfg.saturate,
            compensated=formats.compensated,
            rbits_add=formats.rbits_add,
            rbits_mul=formats.rbits_mul,
        )
    elif type(add_cfg) == SuperNormalFloat:
        return superfp_bmm(
            a,
            b,
            man_add=add_cfg.man,
            exp_add=add_cfg.exp,
            man_mul=mul_cfg.man,
            exp_mul=mul_cfg.exp,
            binades_add=add_cfg.binades,
            binades_mul=mul_cfg.binades,
            rounding=rnd,
            fma=fma,
            saturate=add_cfg.saturate,
        )
    else:  # fixed-point
        return fxp_bmm(
            a,
            b,
            wl_add=add_cfg.wl,
            fl_add=add_cfg.fl,
            wl_mul=mul_cfg.wl,
            fl_mul=mul_cfg.fl,
            symmetric=False,
            rounding=rnd,
            fma=fma,
        )


def float_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 23,
    exp_add: int = 8,
    man_mul: int = 23,
    exp_mul: int = 8,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
    subnormals: bool = True,
    saturate: bool = True,
    compensated: bool = False,
    rbits_add: int = 0,
    rbits_mul: int = 0,
) -> torch.Tensor:
    if cublas_acceleration.enabled and a.is_cuda and b.is_cuda:
        types = match_mac_format_with_cublas_types(
            man_add,
            exp_add,
            man_mul,
            exp_mul,
            rounding,
            fma,
            subnormals,
            saturate,
            cublas_acceleration.fast_mode,
        )
        if types is not None:
            input_type, output_type, compute_type = types
            return cublas_bmm(a, b, input_type, output_type, compute_type, False)

    assert a.shape[-1] == b.shape[-2]
    assert a.device == b.device
    quant_module = get_module(a)
    if rounding == "nearest":
        if not fma:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.float_quantize_nearest_bmm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    subnormals,
                    saturate,
                    compensated,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_nearest_mm(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    subnormals,
                    saturate,
                    compensated,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.float_quantize_nearest_bmm(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    subnormals,
                    saturate,
                    compensated,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_nearest_mm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    subnormals,
                    saturate,
                    compensated,
                )
            else:
                raise Exception("Wrong tensor sizes")
        else:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.float_quantize_nearest_bmm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    subnormals,
                    saturate,
                    compensated,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_nearest_mm_fma(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    subnormals,
                    saturate,
                    compensated,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.float_quantize_nearest_bmm_fma(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    subnormals,
                    saturate,
                    compensated,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_nearest_mm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    subnormals,
                    saturate,
                    compensated,
                )
            else:
                raise Exception("Wrong tensor sizes")
    else:
        if rbits_add <= 0:
            rbits_add = 23 - man_add
        if rbits_mul <= 0:
            rbits_mul == 23 - man_mul
        if not fma:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.float_quantize_stochastic_bmm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    rbits_add,
                    man_mul,
                    exp_mul,
                    rbits_mul,
                    subnormals,
                    saturate,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_stochastic_mm(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    rbits_add,
                    man_mul,
                    exp_mul,
                    rbits_mul,
                    subnormals,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.float_quantize_stochastic_bmm(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    rbits_add,
                    man_mul,
                    exp_mul,
                    rbits_mul,
                    subnormals,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_stochastic_mm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    rbits_add,
                    man_mul,
                    exp_mul,
                    rbits_mul,
                    subnormals,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
        else:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.float_quantize_stochastic_bmm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    rbits_add,
                    subnormals,
                    saturate,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_stochastic_mm_fma(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    rbits_add,
                    subnormals,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.float_quantize_stochastic_bmm_fma(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    rbits_add,
                    subnormals,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.float_quantize_stochastic_mm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    rbits_add,
                    subnormals,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
    return c


def superfp_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    man_add: int = 7,
    exp_add: int = 8,
    man_mul: int = 7,
    exp_mul: int = 8,
    binades_add: int | tuple[int] | tuple[int, int] = 1,
    binades_mul: int | tuple[int] | tuple[int, int] = 1,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
    saturate: bool = True,
) -> torch.Tensor:

    assert a.shape[-1] == b.shape[-2]
    assert a.device == b.device
    quant_module = get_module(a)

    binades_add_l, binades_add_h = normalize_binades(binades_add)
    binades_mul_l, binades_mul_h = normalize_binades(binades_mul)

    if rounding == "nearest":
        if not fma:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.superfp_quantize_nearest_bmm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    binades_add_l,
                    binades_add_h,
                    binades_mul_l,
                    binades_mul_h,
                    saturate,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.superfp_quantize_nearest_mm(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    binades_add_l,
                    binades_add_h,
                    binades_mul_l,
                    binades_mul_h,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.superfp_quantize_nearest_bmm(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    binades_add_l,
                    binades_add_h,
                    binades_mul_l,
                    binades_mul_h,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.superfp_quantize_nearest_mm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    man_mul,
                    exp_mul,
                    binades_add_l,
                    binades_add_h,
                    binades_mul_l,
                    binades_mul_h,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
        else:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.superfp_quantize_nearest_bmm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    man_add,
                    exp_add,
                    binades_add_l,
                    binades_add_h,
                    saturate,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.superfp_quantize_nearest_mm_fma(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    man_add,
                    exp_add,
                    binades_add_l,
                    binades_add_h,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.superfp_quantize_nearest_bmm_fma(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    man_add,
                    exp_add,
                    binades_add_l,
                    binades_add_h,
                    saturate,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.superfp_quantize_nearest_mm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    man_add,
                    exp_add,
                    binades_add_l,
                    binades_add_h,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
    else:
        # TODO: stochastic rounding is not implemented yet
        raise NotImplementedError("SR SuperNormalFloat BMM is not yet implemented")
    return c


def fxp_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    wl_add: int = 16,
    fl_add: int = 8,
    wl_mul: int = 16,
    fl_mul: int = 8,
    symmetric: bool = False,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    fma: bool = True,
) -> torch.Tensor:

    assert a.shape[-1] == b.shape[-2]
    assert a.device == b.device
    quant_module = get_module(a)
    if rounding == "nearest":
        if not fma:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.fixed_point_quantize_nearest_bmm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    wl_add,
                    fl_add,
                    wl_mul,
                    fl_mul,
                    symmetric,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_nearest_mm(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    wl_add,
                    fl_add,
                    wl_mul,
                    fl_mul,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.fixed_point_quantize_nearest_bmm(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    wl_add,
                    fl_add,
                    wl_mul,
                    fl_mul,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_nearest_mm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    wl_add,
                    fl_add,
                    wl_mul,
                    fl_mul,
                    symmetric,
                )
            else:
                raise Exception("Wrong tensor sizes")
        else:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.fixed_point_quantize_nearest_bmm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_nearest_mm_fma(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.fixed_point_quantize_nearest_bmm_fma(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_nearest_mm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            else:
                raise Exception("Wrong tensor sizes")
    else:
        if not fma:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.fixed_point_quantize_stochastic_bmm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_stochastic_mm(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.fixed_point_quantize_stochastic_bmm(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_stochastic_mm(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            else:
                raise Exception("Wrong tensor sizes")
        else:
            if len(a.shape) == 3 and len(b.shape) == 3:
                c = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device)
                quant_module.fixed_point_quantize_stochastic_bmm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[1],
                    b.shape[2],
                    a.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            elif len(a.shape) == 3 and len(b.shape) == 2:
                a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
                c_r = torch.zeros(a_r.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_stochastic_mm_fma(
                    a_r.contiguous(),
                    b.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[0],
                    b.shape[1],
                    a_r.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
            elif len(a.shape) == 4 and len(b.shape) == 4:
                a_r = torch.reshape(
                    a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
                )
                b_r = torch.reshape(
                    b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
                )
                c_r = torch.zeros(
                    a_r.shape[0], a_r.shape[1], b_r.shape[2], device=a.device
                )
                quant_module.fixed_point_quantize_stochastic_bmm_fma(
                    a_r.contiguous(),
                    b_r.contiguous(),
                    c_r.contiguous(),
                    a_r.shape[1],
                    b_r.shape[2],
                    a_r.shape[2],
                    wl_add,
                    fl_add,
                    symmetric,
                )
                c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
            elif len(a.shape) == 2 and len(b.shape) == 2:
                c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
                quant_module.fixed_point_quantize_stochastic_mm_fma(
                    a.contiguous(),
                    b.contiguous(),
                    c.contiguous(),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    wl_add,
                    fl_add,
                    symmetric,
                )
            else:
                raise Exception("Wrong tensor sizes")
    return c


# TODO: create exhaustive tests for the quantizer function
def quantizer(
    forward_number: Number | None = None,
    backward_number: Number | None = None,
    forward_rounding: Literal["nearest", "stochastic"] = "stochastic",
    backward_rounding: Literal["nearest", "stochastic"] = "stochastic",
    clamping_grad_zero: bool = False,
    backward_hooks=[],
):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        forward_number: the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        backward_number: the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        forward_rounding: rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        backward_rounding: rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        clamping_grad_zero: zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        backward_hooks: iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified
    """

    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in [
            "stochastic",
            "nearest",
        ], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None:
            assert isinstance(num, Number)

    if clamping_grad_zero == False:
        if forward_rounding == "nearest":
            if type(forward_number) == BlockFloatingPoint:
                forward_quant = (
                    lambda x, quant_module: quant_module.block_quantize_nearest(
                        x, forward_number.wl, forward_number.dim
                    )
                )
            elif type(forward_number) == FixedPoint:
                forward_quant = (
                    lambda x, quant_module: quant_module.fixed_point_quantize_nearest(
                        x,
                        forward_number.wl,
                        forward_number.fl,
                        forward_number.clamp,
                        forward_number.symmetric,
                    )
                )
            elif type(forward_number) == FloatingPoint:
                forward_quant = (
                    lambda x, quant_module: quant_module.float_quantize_nearest(
                        x,
                        forward_number.man,
                        forward_number.exp,
                        forward_number.subnormals,
                        forward_number.saturate,
                    )
                )
            elif type(forward_number) == SuperNormalFloat:
                fwd_binades_l, fwd_binades_h = normalize_binades(forward_number.binades)
                forward_quant = (
                    lambda x, quant_module: quant_module.superfp_quantize_nearest(
                        x,
                        forward_number.man,
                        forward_number.exp,
                        fwd_binades_l,
                        fwd_binades_h,
                        forward_number.saturate,
                    )
                )
        elif forward_rounding == "stochastic":
            if type(forward_number) == BlockFloatingPoint:
                forward_quant = (
                    lambda x, quant_module: quant_module.block_quantize_stochastic(
                        x, forward_number.wl, forward_number.dim
                    )
                )
            elif type(forward_number) == FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic(
                    x,
                    forward_number.wl,
                    forward_number.fl,
                    forward_number.clamp,
                    forward_number.symmetric,
                )
            elif type(forward_number) == FloatingPoint:
                forward_quant = (
                    lambda x, quant_module: quant_module.float_quantize_stochastic(
                        x,
                        forward_number.man,
                        forward_number.exp,
                        forward_number.subnormals,
                        forward_number.saturate,
                    )
                )
            elif type(forward_number) == SuperNormalFloat:
                # TODO: to implement
                raise NotImplementedError("SR SuperNormalFloat not yet implemented")
    else:
        if type(forward_number) == FixedPoint or forward_number == None:
            assert (
                forward_number == None or forward_number.clamp == True
            ), "must use clamping if zeroing out clamped gradient"
            if forward_rounding == "nearest":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest_mask(
                    x, forward_number.wl, forward_number.fl, forward_number.symmetric
                )
            elif forward_rounding == "stochastic":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic_mask(
                    x, forward_number.wl, forward_number.fl, forward_number.symmetric
                )
        else:
            raise ValueError("zeroing clamping gradient only support fixed point.")

    if backward_rounding == "nearest":
        if type(backward_number) == BlockFloatingPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.block_quantize_nearest(
                    a, backward_number.wl, backward_number.dim
                )
            )
        elif type(backward_number) == FixedPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.fixed_point_quantize_nearest(
                    a,
                    backward_number.wl,
                    backward_number.fl,
                    backward_number.clamp,
                    backward_number.symmetric,
                )
            )
        elif type(backward_number) == FloatingPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.float_quantize_nearest(
                    a,
                    backward_number.man,
                    backward_number.exp,
                    backward_number.subnormals,
                    backward_number.saturate,
                )
            )
        elif type(backward_number) == SuperNormalFloat:
            bwd_binades_l, bwd_binades_h = normalize_binades(backward_number.binades)
            backward_quant = (
                lambda a, quant_module: quant_module.superfp_quantize_nearest(
                    a,
                    backward_number.man,
                    backward_number.exp,
                    bwd_binades_l,
                    bwd_binades_h,
                    backward_number.saturate,
                )
            )
    elif backward_rounding == "stochastic":
        if type(backward_number) == BlockFloatingPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.block_quantize_stochastic(
                    a, backward_number.wl, backward_number.dim
                )
            )
        elif type(backward_number) == FixedPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.fixed_point_quantize_stochastic(
                    a,
                    backward_number.wl,
                    backward_number.fl,
                    backward_number.clamp,
                    backward_number.symmetric,
                )
            )
        elif type(backward_number) == FloatingPoint:
            backward_quant = (
                lambda a, quant_module: quant_module.float_quantize_stochastic(
                    a,
                    backward_number.man,
                    backward_number.exp,
                    backward_number.subnormals,
                    backward_number.saturate,
                )
            )
        elif type(backward_number) == SuperNormalFloat:
            # TODO: to be implemented
            raise NotImplementedError("SR SuperNormalFloat not yet implemented")
    if clamping_grad_zero == False:

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number == None:
                    return x

                quant_module = get_module(x)
                out = forward_quant(x.contiguous(), quant_module)

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        grad_input = backward_quant(
                            grad_output.contiguous(), quant_module
                        )
                else:
                    grad_input = None

                return grad_input

    else:

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number == None:
                    self.mask = torch.zeros_like(x).bool()
                    return x
                else:
                    quant_module = get_module(x)
                    out, mask = forward_quant(x.contiguous(), quant_module)
                    self.mask = mask

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        # grad_output = grad_output.contiguous().masked_fill_(self.mask, 0)
                        for f in backward_hooks:
                            grad_output = f(grad_output)
                        grad_input = backward_quant(
                            grad_output.contiguous(), quant_module
                        ).masked_fill(self.mask.bool(), 0)
                else:
                    grad_input = None

                return grad_input

    return Rounding.apply


def fixed_point_quantize(
    x: torch.Tensor,
    wl: int,
    fl: int,
    clamp: bool = True,
    symmetric: bool = False,
    rounding: Literal["nearest", "stochastic"] = "stochastic",
) -> torch.Tensor:
    """
    Quantize a single precision floating-point tensor into a low-precision fixed-point tensor

    Args:
        x:  the single precision tensor to be quantized
        wl: word length of the fixed-point format being simulated
        fl: fractional length of the fixed-point format being simulated
        clamp: clamp input numbers into representable range. If false, the quantization will only simulate the effect on precision
        symmetric: discard the minimum representable number to make the representable range symmetric
        rounding: rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")

    Returns:
        a quantized fixed-point representation of the input tensor
    """
    assert isinstance(x, torch.Tensor)
    assert rounding in ["stochastic", "nearest"]
    assert_wl_fl(wl, fl)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.fixed_point_quantize_nearest(
            x.contiguous(), wl, fl, clamp, symmetric
        )
    elif rounding == "stochastic":
        out = quant_module.fixed_point_quantize_stochastic(
            x.contiguous(), wl, fl, clamp, symmetric
        )
    return out


def block_quantize(
    x: torch.Tensor,
    wl: int,
    dim: int = -1,
    rounding: Literal["nearest", "stochastic"] = "stochastic",
) -> torch.Tensor:
    """
    Quantize a single precision floating-point tensor into a low-precision block floating-point representation

    Args:
        x:  the single precision tensor to be quantized
        wl: word length of the block floating-point format being simulated
        dim: dimension over which to apply the block floating point representation (-1 applies it to the entire tensor)
        rounding: rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        a quantized low-precision block floating-point representation of the input tensor
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.block_quantize_nearest(x.contiguous(), wl, dim)
    elif rounding == "stochastic":
        out = quant_module.block_quantize_stochastic(x.contiguous(), wl, dim)
    return out


def float_quantize(
    x: torch.Tensor,
    exp: int,
    man: int,
    rounding: Literal["nearest", "stochastic"] = "stochastic",
    subnormals: bool = True,
    saturate: bool = True,
    prng: int = 0,
) -> torch.Tensor:
    """
    Quantize a single precision floating-point tensor into a IEEE-754-like low-precision floating-point tensor

    Args:
        x: the single precision number to be quantized
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, not counting the virtual bit
        rounding: rounding mode, \"stochastic\" or \"nearest\"
        subnormals: if subnormals are supported or not
        saturate: saturate on overflow or use infinities
        prng: number of random bits to use in case of stochastic rounding

    Returns:
        a quantized low-precision floating point representation of the input tensor
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.float_quantize_nearest(
            x.contiguous(), man, exp, subnormals, saturate
        )
    elif rounding == "stochastic":
        if prng == 0:
            prng = 23 - man
        out = quant_module.float_quantize_stochastic(
            x.contiguous(), man, exp, prng, subnormals, saturate
        )
    return out


def binary8_quantize(
    x: torch.Tensor,
    P: int,
    rounding: Literal["nearest", "stochastic", "truncate"] = "nearest",
    overflow_policy: Literal[
        "saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"
    ] = "saturate_maxfloat",
    is_signed: bool = True,
    subnormals: bool = True,
    prng_bits: int = 0,
) -> torch.Tensor:
    """
    Quantize a single precision floating-point tensor into a P3109-compatible one

    Args:
        x: the single precision number(torch.Tensor) to be quantized
        P: number of bits allocated for precision
        is_signed: if subnormals are supported or not
        rounding: the quantization rounding mode
        overflow_policy: overflow handling policy
        subnormals: saturate on overflow or use infinities
        prng_bits: number of bits for the random generator

    Overflow Policies:
        - ``saturate_infty``: Finite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float. Infinite inputs will still map to infinities in this mode.
        - ``saturate_maxfloat``: Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7e/0xfe. This number system has an encoding reserved for infinity (0x7f/0xff).
        - ``saturate_maxfloat2``: Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7f/0xff. This number system does not have an encoding reserved for infinity.

    Returns:
        a quantized low-precision floating point representation of the input tensor
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in [
        "stochastic",
        "nearest",
        "truncate",
    ], "invalid rounding mode, {}".format(rounding)
    assert (
        0 <= prng_bits <= 23 - (P - 1)
    ), "prng_bits should be between 0 and 23 minus the number of mantissa bits"

    quant_module = get_module(x)
    overflow_enum = translate_overflow_policy(quant_module, overflow_policy)
    if rounding == "nearest":
        out = quant_module.binary8_quantize_nearest(
            x.contiguous(), P, is_signed, overflow_enum, subnormals
        )
    elif rounding == "stochastic":
        out = quant_module.binary8_quantize_stochastic(
            x.contiguous(), P, prng_bits, is_signed, overflow_enum, subnormals
        )
    elif rounding == "truncate":
        out = quant_module.binary8_quantize_truncate(
            x.contiguous(), P, is_signed, overflow_enum, subnormals
        )
    return out


def superfp_quantize(
    x: torch.Tensor,
    exp: int,
    man: int,
    binades: int | tuple[int] | tuple[int, int],
    rounding: Literal["stochastic", "nearest"] = "nearest",
    saturate: bool = False,
) -> torch.Tensor:
    """
    Quantize a single precision floating-point tensor into a low-precision supernormal floating-point one

    Args:
        x: the single precision number to be quantized
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, not counting the virtual bit
        binades: number of binades that will be transformed into log range
        rounding: rounding mode, \"stochastic\" or \"nearest\"
        saturate: saturate on overflow or use infinities

    Returns:
        a quantized low-precision supernormal floating-point representation of the input tensor
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    quant_module = get_module(x)
    if rounding == "nearest":
        binades_l, binades_u = normalize_binades(binades)

        out = quant_module.superfp_quantize_nearest(
            x.contiguous(), man, exp, binades_l, binades_u, saturate
        )

    elif rounding == "stochastic":
        # TODO
        raise NotImplementedError("SR SuperNormalFloat not yet implemented")
    return out
