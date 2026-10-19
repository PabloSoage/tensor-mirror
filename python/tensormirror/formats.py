"""Binary floating-point formats handled as integer fields, never as host floats.

A finite value is kept as ``(neg, sig, exp)`` with ``value = (-1)**neg * sig * 2**(exp - mbits)``:
``sig`` is the integer significand (implicit bit included for normals) and ``exp`` the
unbiased exponent, ``emin`` for zeros and subnormals. This mirrors what an RTL datapath
sees, so the reference model and the hardware can be compared field by field.
"""

from dataclasses import dataclass

NUM, INF, NAN = "num", "inf", "nan"


@dataclass(frozen=True)
class Format:
    name: str
    ebits: int
    mbits: int

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
        return self.bias

    @property
    def nan(self) -> int:
        """The NaN pattern tensor cores write: all ones but the sign [CIT MMA-Sim §4.2]."""
        return (1 << (self.width - 1)) - 1

    def inf(self, neg: bool) -> int:
        return (int(neg) << (self.width - 1)) | (((1 << self.ebits) - 1) << self.mbits)


FP16 = Format("fp16", 5, 10)
BF16 = Format("bf16", 8, 7)
FP32 = Format("fp32", 8, 23)


@dataclass(frozen=True)
class Decoded:
    kind: str
    neg: bool
    sig: int
    exp: int


def decode(fmt: Format, bits: int) -> Decoded:
    neg = bool((bits >> (fmt.width - 1)) & 1)
    biased = (bits >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    frac = bits & ((1 << fmt.mbits) - 1)
    if biased == (1 << fmt.ebits) - 1:
        return Decoded(NAN if frac else INF, neg, 0, 0)
    if biased == 0:
        return Decoded(NUM, neg, frac, fmt.emin)
    return Decoded(NUM, neg, (1 << fmt.mbits) | frac, biased - fmt.bias)


def encode(fmt: Format, neg: bool, n: int, e: int, rounding: str) -> int:
    """Round ``(-1)**neg * n * 2**e`` (``n >= 0`` an integer) to ``fmt``.

    ``rounding`` is ``"rz"`` (toward zero) or ``"rne"`` (nearest, ties to even). Overflow
    gives infinity in both modes; for RZ that is what MMA-Sim does (a double to float
    cast), not IEEE RZ, and it is not yet checked on a GPU.
    """
    sign = int(neg) << (fmt.width - 1)
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
        return sign | ((top + fmt.bias) << fmt.mbits) | (q & ((1 << fmt.mbits) - 1))
    return sign | q  # subnormal (top == emin)
