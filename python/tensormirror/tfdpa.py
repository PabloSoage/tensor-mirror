"""Truncated fused dot-product-add (T-FDPA), in integer arithmetic.

The model is the one MMA-Sim derived for NVIDIA mixed-precision ``mma`` instructions
(arXiv 2511.10909, Algorithm 7), written again from the paper with integers only, so
that it is an independent oracle and reads like the datapath it will become:

1. exact, unnormalised products: significands multiplied, exponents added;
2. the products and ``c`` aligned at the largest exponent and each truncated toward
   zero to ``F`` fractional bits, then summed exactly in fixed point;
3. the sum rounded once to the output format.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .formats import INF, NAN, Format, decode, encode


@dataclass(frozen=True)
class TFDPA:
    """One instruction's parameters. ``e_zero`` is the exponent a zero term takes."""

    fa: Format
    fb: Format
    fc: Format
    fd: Format
    L_max: int
    F: int
    rounding: str
    e_zero: int


def t_fdpa(p: TFDPA, a: Sequence[int], b: Sequence[int], c: int, fc: Format | None = None) -> int:
    """``c + sum(a[k] * b[k])`` for one group of at most ``L_max`` terms, as raw bits."""
    fc = fc or p.fc
    da = [decode(p.fa, x) for x in a]
    db = [decode(p.fb, x) for x in b]
    dc = decode(fc, c)
    if dc.kind == NAN or any(x.kind == NAN for x in da + db):
        return p.fd.nan
    infs = set()
    terms = []  # (signed significand, fractional bits, exponent)
    for x, y in zip(da, db, strict=True):
        neg = x.neg != y.neg
        if INF in (x.kind, y.kind):
            if (x.kind != INF and x.sig == 0) or (y.kind != INF and y.sig == 0):
                return p.fd.nan  # inf * 0
            infs.add(neg)
            continue
        sig = x.sig * y.sig
        terms.append((-sig if neg else sig, p.fa.mbits + p.fb.mbits, x.exp + y.exp if sig else p.e_zero))
    if dc.kind == INF:
        infs.add(dc.neg)
    if infs:
        return p.fd.nan if len(infs) == 2 else p.fd.inf(infs.pop())
    terms.append((-dc.sig if dc.neg else dc.sig, fc.mbits, dc.exp if dc.sig else p.e_zero))

    emax = max(e for _, _, e in terms)
    total = 0
    for sig, frac, e in terms:
        shift = e - emax + p.F - frac
        mag = abs(sig) << shift if shift >= 0 else abs(sig) >> -shift  # RZ to F bits
        total += -mag if sig < 0 else mag
    d = encode(p.fd, total < 0, abs(total), emax - p.F, p.rounding)
    return 0 if d == 1 << (p.fd.width - 1) else d  # a zero result is +0


Matrix = Sequence[Sequence[int]]


def mma(p: TFDPA, A: Matrix, B: Matrix, C: Matrix) -> list[list[int]]:
    """``D = A @ B + C`` on raw bits; ``K > L_max`` chains groups through ``D``'s format."""
    m, k, n = len(A), len(A[0]), len(B[0])
    L = min(k, p.L_max)
    D = []
    for i in range(m):
        row = []
        for j in range(n):
            d, fc = C[i][j], p.fc
            for k0 in range(0, k, L):
                d = t_fdpa(p, A[i][k0 : k0 + L], [B[kk][j] for kk in range(k0, k0 + L)], d, fc)
                fc = p.fd
            row.append(d)
        D.append(row)
    return D
