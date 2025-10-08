import torch
from mptorch.quant import binaryK_quantize
import numpy as np
import random
import pytest
from tests.markers import available_devices
from gfloat import RoundMode, round_float, Signedness
from gfloat.formats import format_info_p3109
from tests.quant import bits_to_float, float_to_bits, assert_quant


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("K", [4, 6, 8])
@pytest.mark.parametrize(
    "rounding_mode",
    [
        (RoundMode.TiesToEven, "nearest_even"),
        (RoundMode.TiesToAway, "nearest_away"),
        (RoundMode.TowardPositive, "up"),
        (RoundMode.TowardNegative, "down"),
    ],
)
def test_binaryK_to_gfloat_nearest_even_signed(device, K, rounding_mode):
    for P in range(1, K):
        fi = format_info_p3109(K, P, signedness=Signedness.Signed)
        bias = 2 ** (K - P - 1)
        start_value = 2 ** (1 - bias - P - 3)
        istart_value = float_to_bits(start_value)
        increment = 0x7FFFFFFF & (1 << (22 - (P - 1) - 2))

        vals_to_test = [
            bits_to_float(istart_value + i * increment) for i in range(0, 2 ** (K + 1))
        ]
        vals_to_test_neg = [-x for x in vals_to_test]
        vals_to_test = [vals_to_test, vals_to_test_neg]
        x = torch.tensor(vals_to_test, dtype=torch.float32).to(device)
        gqx = x.clone().detach().to("cpu")
        gqx.apply_(lambda x: round_float(fi, x, rnd=rounding_mode[0]))
        gqx = gqx.to(device)
        qx = binaryK_quantize(x, K, P, rounding=rounding_mode[1])

        print(torch.all(qx == gqx))

        assert torch.all(qx == gqx)
