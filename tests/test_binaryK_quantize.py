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
def test_binaryK_signed_to_gfloat(device, K):
    for P in range(1, K):
        fi = format_info_p3109(K, P, signedness=Signedness.Signed)
        bias = 2 ** (K - P - 1)
        start_value = 2 ** (1 - bias - P - 3)
        istart_value = float_to_bits(start_value)
        increment = 0x7FFFFFFF & (1 << (22 - (P - 1) - 2))
