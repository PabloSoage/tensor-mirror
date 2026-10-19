import dataclasses
import struct

import pytest

from tcmirror.arch import tfdpa
from tcmirror.formats import FP16, FP32, decode, encode
from tcmirror.tfdpa import t_fdpa


def f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def as_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# MMA-Sim §6, Table 6: c = 2**23, products -2**23, -0.5, -0.25, -0.125
@pytest.mark.parametrize(
    ("arch", "expected"),
    [("Volta", 0.0), ("Turing", -0.5), ("Ampere", -0.5), ("Hopper", -0.75), ("Blackwell", -0.75)],
)
def test_discrepancy_example(arch, expected):
    p = tfdpa(arch, "f32.f16.f16.f32")
    a = [f16(x) for x in (-(2.0**13), -0.5, -0.25, -0.125)]
    b = [f16(x) for x in (2.0**10, 1.0, 1.0, 1.0)]
    assert as_f32(t_fdpa(p, a, b, f32(2.0**23))) == expected


def test_volta_truncates_at_23_bits_and_rounds_toward_zero():
    p = tfdpa("Volta", "f32.f16.f16.f32")
    one = f16(1.0)
    # 1 + 2**-23 * (1 + 1/2): the 2**-24 term is lost in alignment, the sum is exact
    a = [one, f16(2.0**-12), f16(2.0**-12), 0]
    b = [one, f16(2.0**-11), f16(2.0**-12), 0]
    assert as_f32(t_fdpa(p, a, b, 0)) == 1.0 + 2.0**-23


@pytest.mark.parametrize(("e_zero", "lost_bits"), [(-21, 0), (-20, 1), (-19, 2)])
def test_zero_c_exponent_truncates_small_products(e_zero, lost_bits):
    # With C = +0 in f16 the zero term sits at e_zero (Volta: -20 [CIT MMA-Sim code]),
    # above a 2**-24 product, so the product loses its low bits. Random bits miss this
    # (H-001), hence a directed case; still to be checked on a V100 (T-1.3).
    p = dataclasses.replace(tfdpa("Volta", "f32.f16.f16.f16"), e_zero=e_zero)
    a, b = f16((1 + 2.0**-10) * 2.0**-12), f16((1 + 3 * 2.0**-10) * 2.0**-12)
    sig = 1025 * 1027  # ends in binary 11, so each lost bit shows
    got = as_f32(t_fdpa(p, [a, 0, 0, 0], [b, 0, 0, 0], 0))
    assert got == (sig >> lost_bits << lost_bits) * 2.0**-44


def test_special_values():
    p = tfdpa("Volta", "f32.f16.f16.f32")
    inf, nan, one = 0x7C00, 0x7E00, f16(1.0)
    assert t_fdpa(p, [inf, 0, 0, 0], [0, 0, 0, 0], 0) == FP32.nan  # inf * 0
    assert t_fdpa(p, [inf, inf, 0, 0], [one, one | 0x8000, 0, 0], 0) == FP32.nan  # inf - inf
    assert t_fdpa(p, [inf, 0, 0, 0], [one, 0, 0, 0], 0) == 0x7F800000
    assert t_fdpa(p, [nan, 0, 0, 0], [one, 0, 0, 0], 0) == FP32.nan
    assert t_fdpa(p, [one, 0, 0, 0], [one | 0x8000, 0, 0, 0], f32(1.0)) == 0  # +0, not -0


@pytest.mark.parametrize("bits", [0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x8001, 0xFBFF])
def test_fp16_roundtrip(bits):
    d = decode(FP16, bits)
    assert encode(FP16, d.neg, d.sig, d.exp - FP16.mbits, "rne") == bits


def test_rne_ties_to_even():
    assert encode(FP16, False, 0b1_0000000000_1, -11, "rne") == f16(1.0)  # 1 + 2**-11: tie, even
    assert encode(FP16, False, 0b1_0000000001_1, -11, "rne") == f16(1.0 + 2.0**-9)
    assert encode(FP16, False, 0b1_0000000000_1, -11, "rz") == f16(1.0)
    assert encode(FP16, False, 65520, 0, "rne") == 0x7C00  # rounds up out of range
    assert encode(FP32, False, 1, -149, "rz") == 1  # smallest subnormal
