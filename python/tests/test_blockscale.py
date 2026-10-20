"""Block-scaled models (ST-FDPA and GST-FDPA) on hand-computed cases."""

import struct

from tensormirror.blockscale import block_scale, gst_fdpa, scale, scale_product, st_fdpa

MX = "m16n8k32.mxf8f6f4.1x.f32.e4m3.e4m3.f32.ue8m0"
NV4 = "m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3"
NV2 = "m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0"
ONE_E4M3, ONE_E2M1 = 0x38, 0x2


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_scale_decoding():
    assert scale("ue8m0", 127) == (1, 0, 0)
    assert scale("ue8m0", 0) == (1, 0, -127)
    assert scale("ue8m0", 0xFF) is None
    assert scale("ue4m3", 0x38) == (8, 3, 0)  # 1.0
    assert scale("ue4m3", 0xB8) == (8, 3, 0)  # the sign bit is ignored
    assert scale("ue4m3", 0x7F) is None and scale("ue4m3", 0xFF) is None
    assert scale("ue4m3", 0x01) == (1, 3, -6)  # 2**-9


def test_st_scales_every_product():
    p = block_scale(MX)
    s = scale_product(scale("ue8m0", 128), scale("ue8m0", 129))  # 2 * 4
    assert st_fdpa(p, [ONE_E4M3] * 32, [ONE_E4M3] * 32, 0, s) == f32(256.0)


def test_st_tiny_products_survive_a_zero_c():
    p = block_scale(MX)
    s = scale_product(scale("ue8m0", 52), scale("ue8m0", 52))  # 2**-150 per product
    assert st_fdpa(p, [ONE_E4M3] * 32, [ONE_E4M3] * 32, 0, s) == f32(2.0**-145)


def test_st_overflow_is_infinity():
    p = block_scale(MX)
    s = scale_product(scale("ue8m0", 254), scale("ue8m0", 254))
    assert st_fdpa(p, [ONE_E4M3] * 32, [ONE_E4M3] * 32, 0, s) == 0x7F800000


def test_gst_aligns_by_the_scale_exponent():
    # 1.0 + 2**-36 * 1.0: the second group's scale exponent is 36 below the first's, so with
    # F = 35 it is truncated away, although its value (2**-36) would fit in f32 next to 1.0
    p = block_scale(NV2)
    a = [ONE_E2M1] + [0] * 31 + [ONE_E2M1] + [0] * 31
    s = [scale("ue8m0", 127), scale("ue8m0", 127 - 36)]
    assert gst_fdpa(p, a, [ONE_E2M1] * 64, 0, s) == f32(1.0)
    s = [scale("ue8m0", 127), scale("ue8m0", 127 - 35)]
    assert gst_fdpa(p, a, [ONE_E2M1] * 64, 0, s) == f32(1.0)  # 1 + 2**-35 is below f32's ulp
    assert gst_fdpa(p, a, [ONE_E2M1] * 64, f32(2.0**-12), [scale("ue8m0", 127 - 35)] * 2) == f32(2.0**-12 + 2.0**-34)


def test_gst_group_sum_is_exact():
    p = block_scale(NV4)
    # 16 products of 6 * 6 = 36 in one group: 576, times the scales 0.5 * 1.0
    s = [scale_product(scale("ue4m3", 0x30), scale("ue4m3", 0x38))] * 4
    a = [0x7] * 16 + [0] * 48
    assert gst_fdpa(p, a, [0x7] * 64, 0, s) == f32(288.0)


def test_gst_nan_scale():
    p = block_scale(NV4)
    s = [scale_product(scale("ue4m3", 0x7F), scale("ue4m3", 0x38))] + [scale("ue4m3", 0x38)] * 3
    assert gst_fdpa(p, [0] * 64, [0] * 64, 0, s) == 0x7FFFFFFF
