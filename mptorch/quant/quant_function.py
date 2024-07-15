import torch
from mptorch import (
    Number,
    FixedPoint,
    FloatingPoint,
    SuperNormalFloat,
    BlockFloatingPoint,
)
from torch.utils.cpp_extension import load
import os
from .cublas import cublas_acceleration


__all__ = [
    "fixed_point_quantize",
    "block_quantize",
    "float_quantize",
    "binary8_quantize",
    "bfloat16_quantize",
    "OverflowPolicy",
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
    "cublas_mm",
    "cublas_bmm",
    "CUBLASMatrixType",
    "CUBLASComputeType",
    "cublas_acceleration"
]

current_path = os.path.dirname(os.path.realpath(__file__))
quant_cpu = load(
    name="quant_cpu",
    sources=[
        os.path.join(current_path, "quant_cpu/pybind_cpu.cpp"),
        os.path.join(current_path, "quant_cpu/quant.cpp"),
        os.path.join(current_path, "quant_cpu/bit_helper.cpp"),
        os.path.join(current_path, "quant_cpu/sim_helper.cpp"),
    ],
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
        extra_ldflags=extra_ldflags
    )
else:
    quant_cuda = quant_cpu


def assert_wl_fl(wl, fl, stage=""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def get_module(x):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module

if torch.cuda.is_available():
    OverflowPolicy  = quant_cuda.OverflowPolicy
else:
    OverflowPolicy = None


if torch.cuda.is_available():
    CUBLASComputeType = quant_cuda.CUBLASComputeType
    CUBLASMatrixType  = quant_cuda.CUBLASMatrixType
else:
    CUBLASComputeType, CUBLASMatrixType = None, None


def cublas_mm(a, b, input_type, output_type, compute_type, pedantic):
    """
    Python wrapper for floating-point cuBLAS GEMM (`cublasGemmEx`). This function
    only accepts binary32 (float32) input and output matrices, but allows intermediate
    casting to other datatypes supported by cuBLAS. Please see allowed type combinations
    in the [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/#cublasgemmex).
    Args:
        - :attr: `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        - :attr: `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        - :attr: `input_type` (CUBLASMatrixType) : intermediate float format for input matrices
        - :attr: `output_type` (CUBLASMatrixType) : intermedtiate float format for the output matrix
        - :attr: `compute_type` (CUBLASComputeType) : accumulator type used by cuBLAS GEMM
        - :attr: `pedantic` (bool) : whethe to hint cuBLAS to use pedantic math or not
    Returns:
        - the result of GEMM (torch.Tensor)
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
        b.shape[1], a.shape[0], # transposed
        device=a.device,
        dtype=dtype[output_type]
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
        pedantic
    )
    return c.t().to(torch.float32)


def cublas_bmm(a, b, input_type, output_type, compute_type, pedantic):
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
        c = torch.zeros(a.shape[0], b.shape[2], a.shape[1], # transposed
                        device=a.device,
                        dtype=dtype[output_type])
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
            pedantic
        )
        c = c.transpose(-2, -1)
    elif len(a.shape) == 3 and len(b.shape) == 2:
        a_r = torch.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
        c_r = torch.zeros(b.shape[1], a_r.shape[0],
                          device=a.device,
                          dtype=dtype[output_type])
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
            pedantic
        )
        c_r = c_r.transpose(-2, -1)
        c = torch.reshape(c_r, (a.shape[0], a.shape[1], b.shape[1]))
    elif len(a.shape) == 4 and len(b.shape) == 4:
        a_r = torch.reshape(
            a, (a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
        )
        b_r = torch.reshape(
            b, (b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
        )
        c_r = torch.zeros(
            a_r.shape[0], b_r.shape[2], a_r.shape[1],
            device=a.device,
            dtype=dtype[output_type]
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
            pedantic
        )
        c_r = c_r.transpose(-2, -1)
        c = torch.reshape(c_r, (a.shape[0], a.shape[1], a.shape[2], b.shape[3]))
    elif len(a.shape) == 2 and len(b.shape) == 2:
        c = torch.zeros(b.shape[1], a.shape[0],
                        device=a.device,
                        dtype=dtype[output_type])
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
            pedantic
        )
        c = c.t()
    else:
        raise Exception("Wrong tensor sizes")
    return c.to(torch.float32)


def match_mac_format_with_cublas_types(
    man_add,
    exp_add,
    man_mul,
    exp_mul,
    rounding,
    fma,
    subnormals,
    saturate,
    fast_mode=None
):
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


def mp_mm(a, b, formats, use_forward=True):
    if use_forward:  # FWD format configuration
        add_cfg, mul_cfg, fma, rnd = (
            formats.fwd_add,
            formats.fwd_mul,
            formats.fwd_fma,
            formats.fwd_rnd,
        )
    else:  # BWD format configuration
        add_cfg, mul_cfg, fma, rnd = (
            formats.bwd_add,
            formats.bwd_mul,
            formats.bwd_fma,
            formats.bwd_rnd,
        )
    if type(formats.fwd_add) == FloatingPoint:
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
        )
    elif type(formats.fwd_add) == SuperNormalFloat:
        return superfp_mm(
            a,
            b,
            man_add=add_cfg.man,
            exp_add=add_cfg.exp,
            man_mul=mul_cfg.man,
            exp_mul=mul_cfg.exp,
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
    a,
    b,
    wl_add=16,
    fl_add=8,
    wl_mul=16,
    fl_mul=8,
    symmetric=False,
    rounding="nearest",
    fma=True,
):
    """
    Mixed-precision fixed-point GEMM with customized formats for the multipliers and accumulators
    Args:
        - :attr: `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        - :attr: `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        - :attr: `wl_add` (int) : word length of the fixed point number being simulated for addition
        - :attr: `fl_add` (int) : fractional length of the fixed point number being simulated for addition
        - :attr: `wl_mul` (int) : word length of the fixed point number being simulated for multiplication
        - :attr: `fl_mul` (int) : fractional length of the fixed point number being simulated for multiplicaiton
        - :attr: `fma` (bool) : use fma operation instead of separate multiply and add (uses the
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
    a,
    b,
    man_add=23,
    exp_add=8,
    man_mul=23,
    exp_mul=8,
    rounding="nearest",
    fma=True,
    subnormals=True,
    saturate=True,
):
    """
    Mixed-precision floating-point GEMM with customized formats for the multipliers and accumulators
    Args:
        - :attr: `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        - :attr: `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        - :attr: `exp_add` (int) : number of bits allocated for exponent in addition result
        - :attr: `man_add` (int) : number of bits allocated for mantissa in addition result, not counting the virtual bit
        - :attr: `exp_mul` (int) : number of bits allocated for exponent in multiplication result
        - :attr: `man_mul` (int) : number of bits allocated for mantissa in multiplication result, not counting the virtual bit
        - :attr: `fma` (bool) : use fma operation instead of separate multiply and add (uses the
        man_add and exp_add parameters for the rounding of the fma results)
        - :attr: `subnormals` (bool): allow the use of subnormal values
        - :attr: `saturate` (bool): saturate results (i.e., clamp values at min/max representable
        in the format instead of outputting infinities)
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
            cublas_acceleration.fast_mode
        )
        if types is not None:
            input_type, output_type, compute_type = types
            return cublas_mm(
                a,
                b,
                input_type,
                output_type,
                compute_type,
                False
            )

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
            )
    else:
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
                man_mul,
                exp_mul,
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
                subnormals,
                saturate,
            )
    return c


def superfp_mm(
    a,
    b,
    man_add=7,
    exp_add=8,
    binades_add=1,
    man_mul=7,
    exp_mul=8,
    binades_mul=1,
    rounding="nearest",
    fma=True,
    saturate=False,
):
    """
    Mixed-precision super normal floating-point GEMM with customized formats for the multipliers and accumulators
    Args:
        - :attr: `a` (torch.Tensor): the input of GEMM, with shape:(M, K)
        - :attr: `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        - :attr: `exp_add` (int) : number of bits allocated for exponent in addition result
        - :attr: `man_add` (int) : number of bits allocated for mantissa in addition result, not counting the virtual bit
        - :attr: `exp_mul` (int) : number of bits allocated for exponent in multiplication result
        - :attr: `man_mul` (int) : number of bits allocated for mantissa in multiplication result, not counting the virtual bit
        - :attr: `fma` (bool) : use fma operation instead of separate multiply and add (uses the
        man_add and exp_add parameters for the rounding of the fma results)
        - :attr: `saturate` (bool): saturate results (i.e., clamp values at min/max representable
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
                binades_add,
                binades_mul,
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
                binades_add,
                saturate,
            )
    else:
        # TODO: stochastic rounding is not implemented yet
        raise NotImplementedError("SR SuperNormalFloat MM is not yet implemented")
    return c


def mp_bmm(a, b, formats, use_forward=True):
    if use_forward:  # FWD format configuration
        add_cfg, mul_cfg, fma, rnd = (
            formats.fwd_add,
            formats.fwd_mul,
            formats.fwd_fma,
            formats.fwd_rnd,
        )
    else:  # BWD format configuration
        add_cfg, mul_cfg, fma, rnd = (
            formats.bwd_add,
            formats.bwd_mul,
            formats.bwd_fma,
            formats.bwd_rnd,
        )
    if type(formats.fwd_add) == FloatingPoint:
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
        )
    elif type(formats.fwd_add) == SuperNormalFloat:
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
    a,
    b,
    man_add=23,
    exp_add=8,
    man_mul=23,
    exp_mul=8,
    rounding="nearest",
    fma=True,
    subnormals=True,
    saturate=True,
):
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
            cublas_acceleration.fast_mode
        )
        if types is not None:
            input_type, output_type, compute_type = types
            return cublas_bmm(
                a,
                b,
                input_type,
                output_type,
                compute_type,
                False
            )
    
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
                )
            else:
                raise Exception("Wrong tensor sizes")
    else:
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
                    man_mul,
                    exp_mul,
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
                    man_mul,
                    exp_mul,
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
                    man_mul,
                    exp_mul,
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
                    man_mul,
                    exp_mul,
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
                    subnormals,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
    return c


def superfp_bmm(
    a,
    b,
    man_add=7,
    exp_add=8,
    man_mul=7,
    exp_mul=8,
    binades_add=1,
    binades_mul=1,
    rounding="nearest",
    fma=True,
    saturate=True,
):

    assert a.shape[-1] == b.shape[-2]
    assert a.device == b.device
    quant_module = get_module(a)
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
                    binades_add,
                    binades_mul,
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
                    binades_add,
                    binades_mul,
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
                    binades_add,
                    binades_mul,
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
                    binades_add,
                    binades_mul,
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
                    binades_add,
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
                    binades_add,
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
                    binades_add,
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
                    binades_add,
                    saturate,
                )
            else:
                raise Exception("Wrong tensor sizes")
    else:
        # TODO: stochastic rounding is not implemented yet
        raise NotImplementedError("SR SuperNormalFloat BMM is not yet implemented")
    return c


def fxp_bmm(
    a,
    b,
    wl_add=16,
    fl_add=8,
    wl_mul=16,
    fl_mul=8,
    symmetric=False,
    rounding="nearest",
    fma=True,
):

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


def quantizer(
    forward_number=None,
    backward_number=None,
    forward_rounding="stochastic",
    backward_rounding="stochastic",
    clamping_grad_zero=False,
    backward_hooks=[],
):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        - :param: forward_number (qtorch.Number, optional) : the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: backward_number (qtorch.Number, optional) : the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: forward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: backward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: clamping_grad_zero (bool) : zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        - :param: backward_hooks (iterable) : iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified (torch.Tensor -> torch.Tensor)
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
                forward_quant = (
                    lambda x, quant_module: quant_module.superfp_quantize_nearest(
                        x,
                        forward_number.man,
                        forward_number.exp,
                        forward_number.binades,
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
                # TODO:
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
            backward_quant = (
                lambda a, quant_module: quant_module.superfp_quantize_nearest(
                    a,
                    backward_number.man,
                    backward_number.exp,
                    backward_number.binades,
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
            # TODO:
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


def fixed_point_quantize(x, wl, fl, clamp=True, symmetric=False, rounding="stochastic"):
    """
    Quantize a single precision floating-point tensor into a low-precision fixed-point tensor

    Args:
        - :param: `x` (torch.Tensor) :  the single precision tensor to be quantized
        - :param: `wl` (int) : word length of the fixed-point format being simulated
        - :param: `fl` (int) : fractional length of the fixed-point format being simulated
        - :param: `clamp` (bool, optional) : clamp input numbers into representable range. if false,
                  the quantization will only simulate the effect on precision
        - :param: `symmetric` (bool, optional) : discard the minimum representable number to make the representable
                  range symmetric
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")

    Returns:
        - a quantized fixed-point representable floating-point tensor (torch.Tensor)
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


def block_quantize(x, wl, dim=-1, rounding="stochastic"):
    """
    Quantize a single precision floating-point tensor into a low-precision block floating-point representation

    Args:
        - :param: `x` (torch.Tensor) :  the single precision tensor to be quantized
        - :param: `wl` (int) : word length of the block floating-point format being simulated
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        - a quantized low-precision block floating-point tensor (torch.Tensor)
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


def float_quantize(x, exp, man, rounding="stochastic", subnormals=True, saturate=True):
    """
    Quantize a single precision Floating Point into low-precision Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit
        - :attr: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"
        - :attr: `subnormals` (bool): if subnormals are supported or not
        - :attr: `saturate` (bool): saturate on overflow or use infinities

    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
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
        out = quant_module.float_quantize_stochastic(
            x.contiguous(), man, exp, subnormals, saturate
        )
    return out

def binary8_quantize(x, P, rounding="nearest", overflow_policy="overflow_maxfloat_ff", is_signed=True, subnormals=True, prng_bits=0):
    """
    Quantize a single precision Floating Point into low-precision Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `P` (int) : number of bits allocated for precision
        - :attr: `is_signed` (bool): if subnormals are supported or not
        - :attr: `rouding` (string): change the mode of rounding
        - :attr: `overflow_policy` (string): change the overflow policy
        - :attr: `subnormals` (bool): saturate on overflow or use infinities
        - :attr: `prng_bits` (int): number of bits for the random generator

    Overflow Policies:
        - "overflow_infty": Finite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float.
                            Infinite inputs will still map to infinities in this mode.
        - "overflow_maxfloat_ext_reals": Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate 
                                         to the maximum float represented by 0x7e/0xfe. This number system has an encoding reserved for infinity (0x7f/0xff).
        - "overflow_maxfloat_reals": Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate 
                                         to the maximum float represented by 0x7f/0xff. This number system does not have an encoding reserved for infinity.


    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest", "truncate"], "invalid rounding mode, {}".format(
        rounding
    )
    assert overflow_policy in ["overflow_infty", "overflow_maxfloat_ext_reals", "overflow_maxfloat_reals"], "invalid overflow policy, {}".format(
        overflow_policy
    )
    assert 0 <= prng_bits <= 23 - (P - 1), "prng_bits should be between 0 and 23 minus the number of mantissa bits"
    saturation_enum = {
        "overflow_infty": OverflowPolicy.OVERLY_INFTY,
        "overflow_maxfloat_ext_reals": OverflowPolicy.OVERFLOW_MAXFLOAT_EXT_REALS,
        "overflow_maxfloat_reals": OverflowPolicy.MAXFLOAT_REALS,
    }[overflow_policy]

    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.binary8_quantize_nearest(
            x.contiguous(), P, is_signed, saturation_enum, subnormals
        )
    elif rounding == "stochastic":
        out = quant_module.binary8_quantize_stochastic(
            x.contiguous(), P, prng_bits, is_signed, saturation_enum, subnormals
        )
    elif rounding == "truncate":
        out = quant_module.binary8_quantize_truncate(
            x.contiguous(), P, is_signed, saturation_enum, subnormals
        )
    return out

def bfloat16_quantize(x, rounding="nearest", prng_bits=0):
    """
    Quantize a single precision Floating Point into bfloat16 Floating Point

    Args:
        - :attr: `x` (torch.Tensor bfloat16_quantize_nearest: the single precision number(torch.Tensor) to be quantized
        - :attr: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"
        - :attr: `prng_bits` (int): number of bits for the random generator

    Returns:
        - a quantized bfloat16 floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"

    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    assert 0 <= prng_bits <= 23 - 7, "prng_bits should be between 0 and 23 minus the number of mantissa bits"

    
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.bfloat16_quantize_nearest(
            x.contiguous()
        )
    elif rounding == "stochastic":
        out = quant_module.bfloat16_quantize_stochastic(
            x.contiguous(), prng_bits
        )
    return out

def superfp_quantize(x, exp, man, binades, rounding="nearest", saturate=False):
    """
    Quantize a single precision Floating Point into low-precision Super Normal Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit
        - :attr: `binades` (int) : number of binades that will be transformed into log range
        - :attr: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"
        - :attr: `saturate` (bool): saturate on overflow or use infinities

    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.superfp_quantize_nearest(x.contiguous(), man, exp, binades, saturate)
    elif rounding == "stochastic":
        # TODO
        raise NotImplementedError("SR SuperNormalFloat not yet implemented")
    return out
