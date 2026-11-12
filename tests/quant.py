"""Helpers shared by the elementwise quantizer tests: float32 words as
integers, for sweeps that step through a format's range word by word."""

import struct

import torch


def bits_to_float(bits):
    """The float32 value whose 32-bit word is ``bits``."""
    s = struct.pack(">I", bits)
    return struct.unpack(">f", s)[0]


def float_to_bits(value):
    """The 32-bit word of ``value`` as a float32, as an unsigned integer."""
    s = struct.pack(">f", value)
    return struct.unpack(">I", s)[0]


def assert_quant(x_arr, expected_arr, quant_fn, device):
    """Assert that ``quant_fn`` maps the float32 tensor built from ``x_arr`` to
    exactly the one built from ``expected_arr``, on ``device``."""
    x = torch.tensor(x_arr, dtype=torch.float32, device=device)
    expected = torch.tensor(expected_arr, dtype=torch.float32, device=device)
    assert expected.equal(quant_fn(x))
