"""Binary floating-point formats handled as integer fields, never as host floats.

A finite value is kept as ``(neg, sig, exp)`` with ``value = (-1)**neg * sig * 2**(exp - mbits)``:
``sig`` is the integer significand (implicit bit included for normals) and ``exp`` the
unbiased exponent, ``emin`` for zeros and subnormals. This mirrors what an RTL datapath
sees, so the reference model and the hardware can be compared field by field.

A format can sit inside a wider register word: ``lsb`` is the position of its lowest bit.
TF32 is the top 19 bits of an f32 word (``lsb = 13``, so the low mantissa bits are simply
dropped); FP4 E2M1 sits in bits 5:2 of a byte and FP6 in bits 5:0, as ``mma`` reads them
on ``sm_120`` [CIT MMA-Sim code, ``decode_uint8_to_fp6_fp4``].
"""

from dataclasses import dataclass

NUM, INF, NAN = "num", "inf", "nan"
IEEE, FN, FINITE = "ieee", "fn", "finite"  # specials: IEEE-like, NaN only (OCP E4M3), none


@dataclass(frozen=True)
class Format:
    name: str
    ebits: int
    mbits: int
    specials: str = IEEE
    lsb: int = 0

    @property
    def bias(self) -> int:
        return (1 << (self.ebits - 1)) - 1

    @property
    def width(self) -> int:
        return 1 + self.ebits + self.mbits

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def emax(self) -> int:
        return self.bias if self.specials == IEEE else self.bias + 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.lsb + self.width - 1)

    @property
    def nan(self) -> int:
        """The NaN word tensor cores write: all ones but the sign [CIT MMA-Sim §4.2]."""
        return self.sign_bit - 1

    def inf(self, neg: bool) -> int:
        return (self.sign_bit if neg else 0) | (((1 << self.ebits) - 1) << (self.mbits + self.lsb))


FP32 = Format("fp32", 8, 23)
FP16 = Format("fp16", 5, 10)
BF16 = Format("bf16", 8, 7)
TF32 = Format("tf32", 8, 10, lsb=13)
E8M13 = Format("e8m13", 8, 13, lsb=10)  # Ada and Hopper FP8 accumulation, written as f32
E5M2 = Format("e5m2", 5, 2)
E4M3 = Format("e4m3", 4, 3, FN)
E3M2 = Format("e3m2", 3, 2, FINITE)
E2M3 = Format("e2m3", 2, 3, FINITE)
E2M1 = Format("e2m1", 2, 1, FINITE, lsb=2)


@dataclass(frozen=True)
class Decoded:
    kind: str
    neg: bool
    sig: int
    exp: int


def decode(fmt: Format, word: int) -> Decoded:
    bits = (word >> fmt.lsb) & ((1 << fmt.width) - 1)
    neg = bool(bits >> (fmt.width - 1))
    biased = (bits >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    frac = bits & ((1 << fmt.mbits) - 1)
    top = biased == (1 << fmt.ebits) - 1
    if fmt.specials == IEEE and top:
        return Decoded(NAN if frac else INF, neg, 0, 0)
    if fmt.specials == FN and top and frac == (1 << fmt.mbits) - 1:
        return Decoded(NAN, neg, 0, 0)
    if biased == 0:
        return Decoded(NUM, neg, frac, fmt.emin)
    return Decoded(NUM, neg, (1 << fmt.mbits) | frac, biased - fmt.bias)


def encode(fmt: Format, neg: bool, n: int, e: int, rounding: str) -> int:
    """Round ``(-1)**neg * n * 2**e`` (``n >= 0`` an integer) to an IEEE-like ``fmt``.

    ``rounding`` is ``"rz"`` (toward zero) or ``"rne"`` (nearest, ties to even). Overflow
    gives infinity in both modes; for RZ that is what MMA-Sim does (a double to float
    cast), not IEEE RZ, and it is not yet checked on a GPU.
    """
    assert fmt.specials == IEEE, "only IEEE-like formats are outputs"
    sign = fmt.sign_bit if neg else 0
    if n == 0:
        return sign
    top = n.bit_length() - 1 + e  # value in [2**top, 2**(top + 1))
    top = max(top, fmt.emin)
    shift = e - top + fmt.mbits  # q = n * 2**shift, in units of the last place
    if shift >= 0:
        q = n << shift
    else:
        q, rem = n >> -shift, n & ((1 << -shift) - 1)
        if rounding == "rne":
            half = 1 << (-shift - 1)
            if rem > half or (rem == half and q & 1):
                q += 1
        elif rounding != "rz":
            raise ValueError(f"unknown rounding {rounding!r}")
    if q >> (fmt.mbits + 1):  # rounding carried into a new binade
        q >>= 1
        top += 1
    if top > fmt.emax:
        return fmt.inf(neg)
    if q >> fmt.mbits:
        q = ((top + fmt.bias) << fmt.mbits) | (q & ((1 << fmt.mbits) - 1))
    return sign | (q << fmt.lsb)  # q alone is a subnormal (top == emin)
