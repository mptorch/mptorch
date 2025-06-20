import torch
from torch.testing import assert_close
from mptorch.quant import cublas_mm, cublas_bmm
from mptorch.quant import CUBLASComputeType as ct, CUBLASMatrixType as mt
from mptorch.quant import float_mm, float_bmm
from mptorch.quant.quant_function import match_mac_format_with_cublas_types
from mptorch.quant import cublas_acceleration
from tests.markers import requires_cuda
import pytest


@requires_cuda
def test_mm_if32_of32_cf32_p():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_cublas = cublas_mm(a, b, mt.F32, mt.F32, ct.F32, True)
    res_mp = float_mm(a, b, 23, 8, 23, 8)
    assert res_cublas.dtype == torch.float32
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-6)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-5)


@requires_cuda
def test_mm_if16_of16_cf16_p():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_cublas = cublas_mm(a, b, mt.F16, mt.F16, ct.F16, True)
    res_mp = float_mm(a, b, 10, 5, 10, 5)
    assert res_cublas.dtype == torch.float32
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-2)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-1)


@requires_cuda
def test_mm_if16_of16_cf16vsf32_p():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_f16 = cublas_mm(a, b, mt.F16, mt.F16, ct.F16, True)
    res_f32 = cublas_mm(a, b, mt.F16, mt.F16, ct.F32, True)
    err_f16 = torch.max(torch.abs(res_f16 - ref) / torch.abs(ref)).item()
    err_f32 = torch.max(torch.abs(res_f32 - ref) / torch.abs(ref)).item()
    assert err_f32 < err_f16


@requires_cuda
def test_mm_if32_of32_ctf32():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_cublas = cublas_mm(a, b, mt.F32, mt.F32, ct.F32_FAST_TF32, False)
    res_mp = float_mm(a, b, 23, 8, 10, 8)
    assert res_cublas.dtype == torch.float32
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-3)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-3)


@requires_cuda
def test_bmm_if32_of32_cf32_p_2_2():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.mm(a, b)
    res_cublas = cublas_bmm(a, b, mt.F32, mt.F32, ct.F32, True)
    res_mp = float_bmm(a, b, 23, 8, 23, 8)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-6)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-5)


@requires_cuda
def test_bmm_if32_of32_cf32_p_3_2():
    a = torch.rand(169, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.bmm(a, b.repeat(169, 1, 1))
    res_cublas = cublas_bmm(a, b, mt.F32, mt.F32, ct.F32, True)
    res_mp = float_bmm(a, b, 23, 8, 23, 8)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-7)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-5)


@requires_cuda
def test_bmm_if32_of32_cf32_p_3_3():
    a = torch.rand(169, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(169, 1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.bmm(a, b)
    res_cublas = cublas_bmm(a, b, mt.F32, mt.F32, ct.F32, True)
    res_mp = float_bmm(a, b, 23, 8, 23, 8)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-7)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-5)


@requires_cuda
def test_bmm_if16_of16_cf16_p_3_3():
    a = torch.rand(169, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(169, 1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.bmm(a, b)
    res_cublas = cublas_bmm(a, b, mt.F16, mt.F16, ct.F16, True)
    res_mp = float_bmm(a, b, 10, 5, 10, 5)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-1)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-1)


@requires_cuda
def test_bmm_ibf16_obf16_cf32_p_3_3():
    a = torch.rand(169, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(169, 1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.bmm(a, b)
    res_cublas = cublas_bmm(a, b, mt.BF16, mt.BF16, ct.F32, True)
    res_mp = float_bmm(a, b, 23, 8, 7, 8)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-1)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-1)


@requires_cuda
def test_bmm_if32_of32_cf32_p_4_4():
    a = torch.rand(9, 5, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(9, 5, 1501, 984, dtype=torch.float32, device="cuda")
    ref = torch.bmm(a.reshape(9 * 5, 277, 1501), b.reshape(9 * 5, 1501, 984)).reshape(
        9, 5, 277, 984
    )
    res_cublas = cublas_bmm(a, b, mt.F32, mt.F32, ct.F32, True)
    res_mp = float_bmm(a, b, 23, 8, 23, 8)
    assert res_cublas.dtype == torch.float32
    assert res_cublas.shape == ref.shape
    assert_close(res_cublas, ref, atol=0.0, rtol=1e-1)
    assert_close(res_cublas, res_mp, atol=0.0, rtol=1e-1)


@requires_cuda
def test_mm_type_error():
    a = torch.rand(277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(1501, 984, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        cublas_mm(a, b, mt.F32, mt.F32, ct.F16, True)


@requires_cuda
def test_bmm_type_error():
    a = torch.rand(8, 277, 1501, dtype=torch.float32, device="cuda")
    b = torch.rand(8, 1501, 984, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        cublas_bmm(a, b, mt.F32, mt.F32, ct.F16, True)


@requires_cuda
def test_cublas_config_for_format():
    assert (
        match_mac_format_with_cublas_types(23, 8, 10, 5, "nearest", True, True, False)
        is None
    )
    assert (
        match_mac_format_with_cublas_types(23, 8, 23, 8, "nearest", False, True, False)
        is None
    )
    assert match_mac_format_with_cublas_types(
        10, 5, 10, 5, "nearest", True, True, False
    ) == (mt.F16, mt.F16, ct.F16)
    assert match_mac_format_with_cublas_types(
        23, 8, 23, 8, "nearest", True, True, False
    ) == (mt.F32, mt.F32, ct.F32)
    assert match_mac_format_with_cublas_types(
        23, 8, 23, 8, "nearest", True, True, False, "f16"
    ) == (mt.F32, mt.F32, ct.F32_FAST_F16)
    assert match_mac_format_with_cublas_types(
        23, 8, 23, 8, "nearest", True, True, False, "bf16"
    ) == (mt.F32, mt.F32, ct.F32_FAST_BF16)
    assert match_mac_format_with_cublas_types(
        23, 8, 23, 8, "nearest", True, True, False, "tf32"
    ) == (mt.F32, mt.F32, ct.F32_FAST_TF32)


@requires_cuda
def test_cublas_override():
    assert not cublas_acceleration.enabled  # check default state
    cublas_acceleration.enable(True, "f16")
    assert cublas_acceleration.enabled
    assert cublas_acceleration.fast_mode == "f16"

    cublas_acceleration.enabled = False
    cublas_acceleration.fast_mode = "tf32"
    with cublas_acceleration(True, "bf16"):
        assert cublas_acceleration.enabled
        assert cublas_acceleration.fast_mode == "bf16"
    assert not cublas_acceleration.enabled
    assert cublas_acceleration.fast_mode == "tf32"
