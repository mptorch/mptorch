import torch
from mptorch.quant import binaryK_quantize
import mptorch
import pytest
from tests.markers import available_devices
from gfloat import RoundMode, round_float, Signedness
from gfloat.formats import format_info_p3109
from tests.quant import bits_to_float, float_to_bits


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K", [4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rounding_mode",
    [
        (RoundMode.TiesToEven, mptorch.number.RoundMode.RNE),
        (RoundMode.TiesToAway, mptorch.number.RoundMode.RNA),
        (RoundMode.TowardPositive, mptorch.number.RoundMode.RU),
        (RoundMode.TowardNegative, mptorch.number.RoundMode.RD),
        (RoundMode.TowardZero, mptorch.number.RoundMode.RZ),
        (RoundMode.ToOdd, mptorch.number.RoundMode.RO),
    ],
)
@pytest.mark.parametrize(
    "signedness", [(Signedness.Signed, True), (Signedness.Unsigned, False)]
)
def test_binaryK_vs_gfloat(device, K, dtype, rounding_mode, signedness):
    for P in range(1, K):
        fi = format_info_p3109(K, P, signedness=signedness[0])
        bias = 2 ** (K - P - 1)
        start_value = 2 ** (1 - bias - P - 3)
        istart_value = float_to_bits(start_value)
        increment = 0x7FFFFFFF & (1 << (22 - (P - 1) - 2))

        vals_to_test = [
            bits_to_float(istart_value + i * increment) for i in range(0, 2 ** (K + 2))
        ]
        if signedness[1]:
            vals_to_test_neg = [-x for x in vals_to_test]
            vals_to_test = [vals_to_test, vals_to_test_neg]

        # We need to cast to the target dtype, then to float32 for gfloat reference
        x = torch.tensor(vals_to_test, dtype=dtype).to(device)

        # gfloat only natively supports float (float32 / float64) in python
        # so we convert to float32 for the reference path
        gqx = x.clone().detach().to("cpu").to(torch.float32)
        gqx.apply_(
            lambda x: round_float(
                fi,
                x,
                rnd=rounding_mode[0],
            )
        )
        gqx = gqx.to(device).to(dtype)
        qx = binaryK_quantize(
            x, K, P, rounding_mode=rounding_mode[1], is_signed=signedness[1]
        )

        assert torch.all(qx == gqx)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "signedness", [(Signedness.Signed, True), (Signedness.Unsigned, False)]
)
def test_binaryK_stochastic(device, K, dtype, signedness):
    P = K - 2 if K > 4 else K - 1

    # 1. Exact Representable Identity & Bounding Guarantees
    # Generate random data strictly within [-0.9, 0.9] (or [0, 0.9]) to avoid overflow logic
    if signedness[1]:
        x_rand = (
            torch.rand(10000, dtype=dtype, device=device) * 1.8 - 0.9
        ).requires_grad_(False)
    else:
        x_rand = (torch.rand(10000, dtype=dtype, device=device) * 0.9).requires_grad_(
            False
        )

    # The number of bits truncated from the IEEE float32 mantissa
    prng_bits = 23 - (P - 1)

    # Get the bounding representable values (Round Down and Round Up)
    q_rd = binaryK_quantize(
        x_rand, K, P, rounding_mode=mptorch.number.RoundMode.RD, is_signed=signedness[1]
    )
    q_ru = binaryK_quantize(
        x_rand, K, P, rounding_mode=mptorch.number.RoundMode.RU, is_signed=signedness[1]
    )

    # Grid points
    x_grid = q_rd.clone()
    q_sr_grid = binaryK_quantize(
        x_grid,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # Property 1: Exact representation identity (grid points shouldn't change)
    assert torch.all(
        q_sr_grid == x_grid
    ), "Stochastic rounding altered an exactly representable grid point!"

    # Quantize the non-representable random points using Stochastic mode
    q_sr_rand = binaryK_quantize(
        x_rand,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # Property 2: Bounding guarantee (SR must pick either RU or RD)
    valid_bounds = (q_sr_rand == q_rd) | (q_sr_rand == q_ru)
    if not torch.all(valid_bounds):
        idx = (~valid_bounds).nonzero(as_tuple=True)[0]
        print(f"FAILED BOUNDS for {len(idx)} elements. First few:")
        for i in idx[:5]:
            print(
                f"x={x_rand[i].item()}, RD={q_rd[i].item()}, RU={q_ru[i].item()}, SR={q_sr_rand[i].item()}"
            )

    assert torch.all(
        valid_bounds
    ), "Stochastic rounding produced a value outside the [RD, RU] bounds!"

    # 3. Statistical Unbiasedness
    # Test a specific non-representable scalar value falling nicely inside the dynamic range
    test_val = 0.333333
    N_samples = 1_000_000
    x_large = torch.full((N_samples,), test_val, dtype=dtype, device=device)

    q_sr_large = binaryK_quantize(
        x_large,
        K,
        P,
        prng_bits=prng_bits,
        rounding_mode=mptorch.number.RoundMode.SR,
        is_signed=signedness[1],
    )

    # The expected value of Q_SR(x) should closely approximate x.
    mean_val = q_sr_large.mean().item()

    # Tolerance based on generous confidence interval
    tolerance = 0.05

    assert (
        abs(mean_val - test_val) < tolerance
    ), f"Stochastic rounding expectation biased! Expected {test_val}, got mean {mean_val}"
