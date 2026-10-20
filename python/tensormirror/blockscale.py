"""Block-scaled fused dot-product-add (RTX Blackwell ``mma ... .block_scale``), in integers.

Two models, both MMA-Sim's [CIT MMA-Sim code, ``arithmetic/fdpa.py`` and ``nv_ptx/sim.py``,
commit 54294d4], written again with integers only:

- **ST-FDPA** (``kind::mxf8f6f4``, one scale per 32 elements): T-FDPA in which each exact
  product is also multiplied by its two scale factors before the alignment.
- **GST-FDPA** (``kind::mxf4`` and ``kind::mxf4nvf4``, packed E2M1): the products of each
  group of ``G`` elements are summed exactly first; a group's sum times its scale is then
  aligned by the scale's exponent alone, not by the value's, truncated to ``F`` fractional
  bits and summed with ``c``. A group whose products are all zero, or whose scale is zero,
  takes ``e_zero``, and so does a zero ``c``.

Scale factors are unsigned: UE8M0 is ``2**(x - 127)`` with 0xFF a NaN; UE4M3 is E4M3 with
the sign bit ignored, as MMA-Sim does (H-010, prediction 4).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .formats import E2M1, E2M3, E3M2, E4M3, E5M2, FINITE, FP32, INF, NAN, Format, decode, encode

E2M1_PACKED = Format("e2m1", 2, 1, FINITE)  # kind::mxf4/mxf4nvf4: a bare nibble
ELEMENT = {"e4m3": E4M3, "e5m2": E5M2, "e3m2": E3M2, "e2m3": E2M3, "e2m1": E2M1}

Scale = tuple[int, int, int] | None  # (significand, fractional bits, exponent); None is NaN


def scale(stype: str, x: int) -> Scale:
    x &= 0xFF
    if stype == "ue8m0":
        return None if x == 0xFF else (1, 0, x - 127)
    if stype == "ue4m3":
        d = decode(E4M3, x & 0x7F)
        return None if d.kind == NAN else (d.sig, E4M3.mbits, d.exp)
    raise ValueError(f"unknown scale type {stype!r}")


def scale_product(sa: Scale, sb: Scale) -> Scale:
    if sa is None or sb is None:
        return None
    return sa[0] * sb[0], sa[1] + sb[1], sa[2] + sb[2]


@dataclass(frozen=True)
class BlockScale:
    """One instruction. ``blocks`` scale factors per row of A (and column of B)."""

    kind: str  # "st" or "gst"
    fa: Format
    fb: Format
    stype: str
    K: int
    blocks: int
    F: int
    e_zero: int
    G: int = 0  # gst only


def st_fdpa(p: BlockScale, a: Sequence[int], b: Sequence[int], c: int, s: Scale) -> int:
    """``c + s * sum(a[k] * b[k])`` with one scale ``s`` (= SF_A * SF_B), as f32 bits."""
    da = [decode(p.fa, x) for x in a]
    db = [decode(p.fb, x) for x in b]
    dc = decode(FP32, c)
    if s is None or dc.kind == NAN or any(x.kind == NAN for x in da + db):
        return FP32.nan
    ssig, sfrac, sexp = s
    infs = set()
    terms = []  # (signed significand, fractional bits, exponent)
    for x, y in zip(da, db, strict=True):
        neg = x.neg != y.neg
        if INF in (x.kind, y.kind):
            if (x.kind != INF and x.sig == 0) or (y.kind != INF and y.sig == 0):
                return FP32.nan  # inf * 0
            infs.add(neg)
            continue
        sig = x.sig * y.sig * ssig
        terms.append((-sig if neg else sig, p.fa.mbits + p.fb.mbits + sfrac,
                      x.exp + y.exp + sexp if sig else p.e_zero))
    if dc.kind == INF:
        infs.add(dc.neg)
    if infs:
        return FP32.nan if len(infs) == 2 else FP32.inf(infs.pop())
    terms.append((-dc.sig if dc.neg else dc.sig, FP32.mbits, dc.exp if dc.sig else p.e_zero))
    return _fused_sum(p.F, terms)


def gst_fdpa(p: BlockScale, a: Sequence[int], b: Sequence[int], c: int, s: Sequence[Scale]) -> int:
    """``c + sum_j s[j] * (a . b over block j)``, groups of ``p.G`` summed exactly first."""
    dc = decode(FP32, c)
    if dc.kind == NAN or any(x is None for x in s):
        return FP32.nan
    if dc.kind == INF:
        return FP32.inf(dc.neg)  # E2M1 has no infinities and the scales are finite
    da = [decode(p.fa, x) for x in a]
    db = [decode(p.fb, x) for x in b]
    emin, frac = p.fa.emin + p.fb.emin, p.fa.mbits + p.fb.mbits
    block = len(a) // len(s)
    terms = []
    for g0 in range(0, len(a), p.G):
        prods = [(x.sig * y.sig << (x.exp + y.exp - emin), x.neg != y.neg)
                 for x, y in zip(da[g0 : g0 + p.G], db[g0 : g0 + p.G], strict=True)]
        total = sum(-m if neg else m for m, neg in prods)  # in units of 2**(emin - frac)
        ssig, sfrac, sexp = s[g0 // block]
        # value = total * ssig * 2**(sexp + emin - frac - sfrac), aligned by the scale's sexp
        zero = ssig == 0 or all(m == 0 for m, _ in prods)  # a sum that cancels keeps sexp
        terms.append((total * ssig, frac + sfrac - emin, p.e_zero if zero else sexp))
    terms.append((-dc.sig if dc.neg else dc.sig, FP32.mbits, dc.exp if dc.sig else p.e_zero))
    return _fused_sum(p.F, terms)


def _fused_sum(F: int, terms: list[tuple[int, int, int]]) -> int:
    """Align at the largest exponent, truncate each term to ``F`` bits, sum, RZ to f32.

    A term ``(sig, frac, e)`` is ``sig * 2**(e - frac)``, aligned by ``e``."""
    emax = max(e for _, _, e in terms)
    total = 0
    for sig, frac, e in terms:
        shift = e - emax + F - frac
        mag = abs(sig) << shift if shift >= 0 else abs(sig) >> -shift  # RZ to F bits
        total += -mag if sig < 0 else mag
    d = encode(FP32, total < 0, abs(total), emax - F, "rz")
    return 0 if d == FP32.sign_bit else d  # a zero result is +0


def block_scale(inst: str, measured: bool = True) -> BlockScale:
    """``inst`` as the harness names it, e.g. ``m16n8k32.mxf8f6f4.1x.f32.e4m3.e2m1.f32.ue8m0``.

    ``measured = False`` reads E2M1 in ``mxf8f6f4`` as MMA-Sim does (bits 5:2)."""
    shape, kind, vec, _, a, b, _, stype = inst.split(".")
    K = int(shape.split("k")[1])
    blocks = int(vec[0])
    if kind == "mxf8f6f4":
        fa, fb = ELEMENT[a], ELEMENT[b]
        if measured:  # E2M1 is read as E2M3 in bits 5:0, as without scales [MED] H-007
            fa = E2M3 if a == "e2m1" else fa
            fb = E2M3 if b == "e2m1" else fb
        return BlockScale("st", fa, fb, stype, K, blocks, F=25, e_zero=-133)
    return BlockScale("gst", E2M1_PACKED, E2M1_PACKED, stype, K, blocks, F=35, e_zero=-139, G=16)


Matrix = Sequence[Sequence[int]]


def mma(p: BlockScale, A: Matrix, B: Matrix, C: Matrix, SFA: Matrix, SFB: Matrix) -> list[list[int]]:
    """``D = (A * SFA) @ (B * SFB) + C`` on raw bits; ``SFA`` is M x blocks, ``SFB`` blocks x N."""
    cols = [[row[j] for row in B] for j in range(len(B[0]))]
    out = []
    for i, arow in enumerate(A):
        line = []
        for j, bcol in enumerate(cols):
            s = [scale_product(scale(p.stype, SFA[i][k]), scale(p.stype, SFB[k][j])) for k in range(p.blocks)]
            if p.kind == "st":
                line.append(st_fdpa(p, arow, bcol, C[i][j], s[0]))
            else:
                line.append(gst_fdpa(p, arow, bcol, C[i][j], s))
        out.append(line)
    return out
