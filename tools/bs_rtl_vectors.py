"""Test vectors for the block-scaled RTL (rtl/tfdpa.sv with ST = 1, rtl/gst_fdpa.sv) from
python/tensormirror/blockscale.py. One case per line, hex:

    a[0..K-1] b[0..K-1] c sfa[0..blocks-1] sfb[0..blocks-1] d

The elements are bytes for kind::mxf8f6f4 (ST) and nibbles for kind::mxf4/mxf4nvf4 (GST).
The cases come in thirds, as in tools/bs_cases.py: everything raw (NaN, zero and extreme
scales included), raw elements with scales near 1, and elements, scales and c near 1; then
cancelling sums and the directed cases.

    python tools/bs_rtl_vectors.py --inst m16n8k32.mxf8f6f4.1x.f32.e4m3.e2m1.f32.ue8m0 --cases 100000 --out v.txt
"""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bs_cases import SCALE_ONE, near_one, near_one_c, near_one_scale, parse  # noqa: E402

from tensormirror.blockscale import block_scale, gst_fdpa, scale, scale_product, st_fdpa  # noqa: E402
from tensormirror.formats import FP32, decode, encode  # noqa: E402


def model(p, ins, a, b, c, sfa, sfb) -> int:
    s = [scale_product(scale(p.stype, x), scale(p.stype, y)) for x, y in zip(sfa, sfb, strict=True)]
    if p.kind == "st":
        return st_fdpa(p, a, b, c, s[0])
    return gst_fdpa(p, a, b, c, s)


def element(rng: random.Random, ins: dict, t: str) -> int:
    x = near_one(rng, ins, t)
    return x & 0xF if ins["packed"] else x


def cases(p, ins: dict, n: int, seed: int):
    rng = random.Random(seed)
    K, nb, st = ins["K"], ins["blocks"], ins["stype"]
    ebits = 4 if ins["packed"] else 8
    third = n // 3
    for i in range(n):
        a = [rng.getrandbits(ebits) for _ in range(K)]
        b = [rng.getrandbits(ebits) for _ in range(K)]
        c = rng.getrandbits(32)
        sfa = [rng.getrandbits(8) for _ in range(nb)]
        sfb = [rng.getrandbits(8) for _ in range(nb)]
        if i >= third:
            sfa = [near_one_scale(rng, st) for _ in range(nb)]
            sfb = [near_one_scale(rng, st) for _ in range(nb)]
        if i >= 2 * third:
            a = [element(rng, ins, ins["ta"]) for _ in range(K)]
            b = [element(rng, ins, ins["tb"]) for _ in range(K)]
            c = near_one_c(rng)
            if i % 2:  # c = -(the scaled sum): what is left is what the alignment changed
                dec = decode(FP32, model(p, ins, a, b, 0, sfa, sfb))
                if dec.kind == "num" and dec.sig:
                    c = encode(FP32, not dec.neg, dec.sig, dec.exp - FP32.mbits, "rne")
        yield a, b, c, sfa, sfb


def directed(p, ins: dict) -> list:
    """(name, a, b, c, sfa, sfb): the group-zero rule of GST-FDPA and the scale specials."""
    K, nb, st = ins["K"], ins["blocks"], ins["stype"]
    one = 0x2 if ins["packed"] else None  # E2M1 1.0 as a bare nibble
    out = []
    if ins["packed"]:
        big = 254 if st == "ue8m0" else 0x7E  # the largest scale
        unit = SCALE_ONE[st]
        g = 16
        zero_g = [0] * g
        cancel_g = [one, one | 0x8] * (g // 2)  # +1 -1 ... : sums to exactly 0
        rest = [0] * (K - g)
        ones_b = [one] * K
        # group 0 cancels: it keeps its scale's exponent, so emax is huge and c is truncated
        out.append(("group_cancels_keeps_scale_exp", cancel_g + rest, ones_b, 0x3F800000,
                    [big] + [unit] * (nb - 1), [big] + [unit] * (nb - 1)))
        # group 0 has only zero products: it takes e_zero, and c survives
        out.append(("group_all_zero_takes_e_zero", zero_g + rest, ones_b, 0x3F800000,
                    [big] + [unit] * (nb - 1), [big] + [unit] * (nb - 1)))
        if st == "ue4m3":
            out.append(("zero_scale_takes_e_zero", [one] * K, ones_b, 0x3F800000, [0x00] * nb, [unit] * nb))
            out.append(("ue4m3_sign_ignored", [one] * K, ones_b, 0, [0xB8] * nb, [0x38] * nb))
    out.append(("nan_scale", [0] * K, [0] * K, 0, [0xFF] * nb, [SCALE_ONE[st]] * nb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ins = parse(args.inst)
    p = block_scale(args.inst)
    rows = list(cases(p, ins, args.cases, args.seed))
    rows += [(a, b, c, sa, sb) for _, a, b, c, sa, sb in directed(p, ins)]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for a, b, c, sfa, sfb in rows:
            d = model(p, ins, a, b, c, sfa, sfb)
            f.write(" ".join(f"{x:x}" for x in [*a, *b, c, *sfa, *sfb, d]) + "\n")
    print(f"{len(rows)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
