import torch
from mptorch.quant import superfp_quantize_v2
from mptorch.number import RoundMode
import pytest
from tests.markers import available_devices


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE])
@pytest.mark.parametrize(
    "test_data",
    [
        [
            torch.tensor(
                [
                    float("inf"),
                    -1.75,
                    5.0,
                    -(2**-18),
                    2**-6 * (1 + 2**-1 + 2**-2 + 2**-5),
                    0.0,
                    2**-19,
                    2**-2 * (1 + 2**-3),
                    -(2**-1) * (1 + 2**-3 + 2**-5),
                ]
            ),
            torch.tensor(
                [
                    1.5,
                    -1.5,
                    1.5,
                    -(2**-18),
                    2**-5,
                    0.0,
                    0.0,
                    2**-2,
                    -(2**-1) * (1 + 2**-2),
                ]
            ),
            torch.tensor(
                [
                    float("inf"),
                    -float("inf"),
                    float("inf"),
                    -(2**-18),
                    2**-5,
                    0.0,
                    0.0,
                    2**-2,
                    -(2**-1) * (1 + 2**-2),
                ]
            ),
        ],
    ],
)
def test_binary6p3b4(test_data, device, mode):
    x = test_data[0].to(device)
    expected_saturate = test_data[1].to(device)
    expected_overflow = test_data[2].to(device)
    def quant_binary6p3b4_saturate(x):
        return superfp_quantize_v2(
            x, 3, 2, (4, 0), saturate=True, rounding_mode=mode, bias=7
        )
    def quant_binary6p3b4_overflow(x):
        return superfp_quantize_v2(
            x, 3, 2, (4, 0), saturate=False, rounding_mode=mode, bias=7
        )

    actual_saturate = quant_binary6p3b4_saturate(x)
    actual_overflow = quant_binary6p3b4_overflow(x)

    assert (expected_saturate == actual_saturate).all()
    assert (expected_overflow == actual_overflow).all()
