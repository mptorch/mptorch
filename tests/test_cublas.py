import torch
from mptorch.quant import cublas_mm
from mptorch.quant import CUBLASComputeType as ct, CUBLASMatrixType as mt

def no_cuda():
    return not torch.cuda.is_available()

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def test_mm_if32_of32_cf32_p():
    if no_cuda():
        return

    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res = cublas_mm(a, b, mt.F32, mt.F32, ct.F32, True)
    assert res.dtype == torch.float32
    err = torch.max(torch.abs(res - ref) / torch.abs(ref)).item()
    assert err < 1e-7

def test_mm_if16_of16_cf16_p():
    if no_cuda():
        return
    
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res = cublas_mm(a, b, mt.F16, mt.F16, ct.F16, True)
    assert res.dtype == torch.float32
    err = torch.max(torch.abs(res - ref) / torch.abs(ref)).item()
    assert err < 1e-2

def test_mm_if16_of16_cf16vsf32_p():
    if no_cuda():
        return
    
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_f16 = cublas_mm(a, b, mt.F16, mt.F16, ct.F16, True)
    res_f32 = cublas_mm(a, b, mt.F16, mt.F16, ct.F32, True)
    err_f16 = torch.max(torch.abs(res_f16 - ref) / torch.abs(ref)).item()
    err_f32 = torch.max(torch.abs(res_f32 - ref) / torch.abs(ref)).item()
    assert err_f32 < err_f16

def test_mm_if32_of32_ctf32():
    if no_cuda():
        return
    
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res = cublas_mm(a, b, mt.F32, mt.F32, ct.FAST_TF32, False)
    assert res.dtype == torch.float32
    err = torch.max(torch.abs(res - ref) / torch.abs(ref)).item()
    assert err < 1e-3