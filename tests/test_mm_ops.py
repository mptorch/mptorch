import mptorch.quant as qt
import pytest
import torch
from torch.testing import assert_close

from mptorch.quant.mm_ops import has_unified_mm_ops
from tests.markers import available_devices


def test_unified_mm_ops_registered():
    assert has_unified_mm_ops()


@pytest.mark.parametrize("device", available_devices)
def test_float_mm_parity_default(device):
    man, exp = 22, 8
    a = torch.randn(32, 48, device=device)
    b = torch.randn(48, 64, device=device)

    legacy = qt.float_mm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    if has_unified_mm_ops():
        unified = qt.float_mm_v2(
            a,
            b,
            man_add=man,
            exp_add=exp,
            man_mul=man,
            exp_mul=exp,
            rounding="RNE",
            fma=False,
            subnormals=True,
            saturate=False,
            compensated=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_bmm_parity_default(device):
    man, exp = 22, 8
    a = torch.randn(5, 32, 48, device=device)
    b = torch.randn(5, 48, 64, device=device)

    legacy = qt.float_bmm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    if has_unified_mm_ops():
        unified = qt.float_bmm_v2(
            a,
            b,
            man_add=man,
            exp_add=exp,
            man_mul=man,
            exp_mul=exp,
            rounding="RNE",
            fma=False,
            subnormals=True,
            saturate=False,
            compensated=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_bmm_parity_3x2(device):
    man, exp = 22, 8
    a = torch.randn(4, 16, 24, device=device)
    b = torch.randn(24, 20, device=device)

    legacy = qt.float_bmm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    if has_unified_mm_ops():
        unified = qt.float_bmm_v2(
            a,
            b,
            man_add=man,
            exp_add=exp,
            man_mul=man,
            exp_mul=exp,
            rounding="RNE",
            fma=False,
            subnormals=True,
            saturate=False,
            compensated=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_bmm_parity_4x4(device):
    man, exp = 22, 8
    a = torch.randn(2, 3, 16, 24, device=device)
    b = torch.randn(2, 3, 24, 20, device=device)

    legacy = qt.float_bmm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    if has_unified_mm_ops():
        unified = qt.float_bmm_v2(
            a,
            b,
            man_add=man,
            exp_add=exp,
            man_mul=man,
            exp_mul=exp,
            rounding="RNE",
            fma=False,
            subnormals=True,
            saturate=False,
            compensated=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_bmm_parity_3x2(device):
    a = torch.randn(4, 16, 24, device=device)
    b = torch.randn(24, 20, device=device)

    legacy = qt.superfp_bmm(
        a,
        b,
        man_add=7,
        exp_add=8,
        man_mul=7,
        exp_mul=8,
        binades_add=1,
        binades_mul=1,
        rounding="RNE",
        fma=False,
        saturate=False,
    )
    if has_unified_mm_ops():
        from mptorch.quant.quant_function import normalize_binades

        binades_l, binades_u = normalize_binades(1)
        unified = qt.superfp_bmm_v2(
            a,
            b,
            man_add=7,
            exp_add=8,
            man_mul=7,
            exp_mul=8,
            binades_add_l=binades_l,
            binades_add_u=binades_u,
            binades_mul_l=binades_l,
            binades_mul_u=binades_u,
            man_fma=7,
            exp_fma=8,
            binades_fma_l=binades_l,
            binades_fma_u=binades_u,
            saturate=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_bmm_parity_4x4(device):
    a = torch.randn(2, 3, 16, 24, device=device)
    b = torch.randn(2, 3, 24, 20, device=device)

    legacy = qt.superfp_bmm(
        a,
        b,
        man_add=7,
        exp_add=8,
        man_mul=7,
        exp_mul=8,
        binades_add=1,
        binades_mul=1,
        rounding="RNE",
        fma=False,
        saturate=False,
    )
    if has_unified_mm_ops():
        from mptorch.quant.quant_function import normalize_binades

        binades_l, binades_u = normalize_binades(1)
        unified = qt.superfp_bmm_v2(
            a,
            b,
            man_add=7,
            exp_add=8,
            man_mul=7,
            exp_mul=8,
            binades_add_l=binades_l,
            binades_add_u=binades_u,
            binades_mul_l=binades_l,
            binades_mul_u=binades_u,
            man_fma=7,
            exp_fma=8,
            binades_fma_l=binades_l,
            binades_fma_u=binades_u,
            saturate=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_fxp_bmm_parity_3x2(device):
    a = torch.randn(4, 16, 24, device=device)
    b = torch.randn(24, 20, device=device)

    legacy = qt.fxp_bmm(
        a,
        b,
        wl_add=16,
        fl_add=8,
        wl_mul=16,
        fl_mul=8,
        symmetric=False,
        rounding="RNE",
        fma=False,
    )
    if has_unified_mm_ops():
        unified = qt.fxp_bmm_v2(
            a,
            b,
            wl_add=16,
            fl_add=8,
            wl_mul=16,
            fl_mul=8,
            wl_fma=16,
            fl_fma=8,
            rounding="RNE",
            symmetric=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_fxp_bmm_parity_4x4(device):
    a = torch.randn(2, 3, 16, 24, device=device)
    b = torch.randn(2, 3, 24, 20, device=device)

    legacy = qt.fxp_bmm(
        a,
        b,
        wl_add=16,
        fl_add=8,
        wl_mul=16,
        fl_mul=8,
        symmetric=False,
        rounding="RNE",
        fma=False,
    )
    if has_unified_mm_ops():
        unified = qt.fxp_bmm_v2(
            a,
            b,
            wl_add=16,
            fl_add=8,
            wl_mul=16,
            fl_mul=8,
            wl_fma=16,
            fl_fma=8,
            rounding="RNE",
            symmetric=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_superfp_mm_parity(device):
    a = torch.randn(16, 24, device=device)
    b = torch.randn(24, 20, device=device)

    legacy = qt.superfp_mm(
        a,
        b,
        man_add=7,
        exp_add=8,
        man_mul=7,
        exp_mul=8,
        binades_add=1,
        binades_mul=1,
        rounding="RNE",
        fma=False,
        saturate=False,
    )
    if has_unified_mm_ops():
        from mptorch.quant.quant_function import normalize_binades

        binades_l, binades_u = normalize_binades(1)
        unified = qt.superfp_mm_v2(
            a,
            b,
            man_add=7,
            exp_add=8,
            man_mul=7,
            exp_mul=8,
            binades_add_l=binades_l,
            binades_add_u=binades_u,
            binades_mul_l=binades_l,
            binades_mul_u=binades_u,
            man_fma=7,
            exp_fma=8,
            binades_fma_l=binades_l,
            binades_fma_u=binades_u,
            saturate=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_fxp_mm_parity(device):
    a = torch.randn(16, 24, device=device)
    b = torch.randn(24, 20, device=device)

    legacy = qt.fxp_mm(
        a,
        b,
        wl_add=16,
        fl_add=8,
        wl_mul=16,
        fl_mul=8,
        symmetric=False,
        rounding="RNE",
        fma=False,
    )
    if has_unified_mm_ops():
        unified = qt.fxp_mm_v2(
            a,
            b,
            wl_add=16,
            fl_add=8,
            wl_mul=16,
            fl_mul=8,
            wl_fma=16,
            fl_fma=8,
            rounding="RNE",
            symmetric=False,
            use_fma=False,
        )
        assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_mm_v2_compensated(device):
    if not has_unified_mm_ops():
        pytest.skip("unified MM ops not registered")
    man, exp = 22, 8
    a = torch.randn(16, 24, device=device)
    b = torch.randn(24, 20, device=device)
    legacy = qt.float_mm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=True,
    )
    unified = qt.float_mm_v2(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=True,
    )
    assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_mm_v2_fma(device):
    if not has_unified_mm_ops():
        pytest.skip("unified MM ops not registered")
    man, exp = 22, 8
    a = torch.randn(16, 24, device=device)
    b = torch.randn(24, 20, device=device)
    legacy = qt.float_mm(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=True,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    unified = qt.float_mm_v2(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="RNE",
        fma=True,
        subnormals=True,
        saturate=False,
        compensated=False,
    )
    assert_close(legacy, unified, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_float_mm_v2_stochastic(device):
    if not has_unified_mm_ops():
        pytest.skip("unified MM ops not registered")
    man, exp = 22, 8
    a = torch.randn(16, 24, device=device)
    b = torch.randn(24, 20, device=device)
    rbits_add = 23 - man
    rbits_mul = 23 - man
    unified = qt.float_mm_v2(
        a,
        b,
        man_add=man,
        exp_add=exp,
        man_mul=man,
        exp_mul=exp,
        rounding="SR",
        fma=False,
        subnormals=True,
        saturate=False,
        compensated=False,
        rbits_add=rbits_add,
        rbits_mul=rbits_mul,
    )
    assert unified.shape == (a.shape[0], b.shape[1])
