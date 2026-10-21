"""Test vectors for rtl/tfdpa_f16.sv from the integer model.

One case per line, hex: a[0..L-1] b[0..L-1] c d. Three kinds of case, interleaved:

- raw:     uniform random bit patterns (NaN, infinities, subnormals all appear);
- scaled:  finite f16 operands with exponents in a narrow window and random signs, so
           the alignment shifts are short and every bit of the sum matters;
- cancel:  as scaled, with c set to minus the model's own sum, so the result is a few
           units of the last place: this exercises the leading-one search;
and then the directed cases of tools/gpu_cases.py.

    python tools/rtl_vectors.py --variant f32.f16.f16.f32 --cases 200000 --seed 1 --out build/v.txt
"""

import argparse
import dataclasses
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gpu_cases import c_bits, directed  # noqa: E402

from tensormirror.arch import PTX_TYPES, tfdpa  # noqa: E402
from tensormirror.formats import FN, FP16, IEEE, decode, encode  # noqa: E402
from tensormirror.tfdpa import t_fdpa  # noqa: E402


def f16_scaled(rng: random.Random, lo: int, hi: int) -> int:
    e = rng.randint(lo, hi)  # biased exponent
    return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)


def f32_scaled(rng: random.Random, lo: int, hi: int) -> int:
    return (rng.getrandbits(1) << 31) | (rng.randint(lo, hi) << 23) | rng.getrandbits(23)


def scaled_word(rng: random.Random, fmt, lo: int, hi: int) -> int:
    """A finite value of fmt with a random sign and mantissa and a biased exponent in [lo, hi]."""
    top = (1 << fmt.ebits) - 1
    e = min(max(rng.randint(lo, hi), 0), top - 1 if fmt.specials == IEEE else top)
    m = rng.getrandbits(fmt.mbits)
    if fmt.specials == FN and e == top and m == (1 << fmt.mbits) - 1:
        m -= 1  # all ones is NaN
    v = (rng.getrandbits(1) << (fmt.ebits + fmt.mbits)) | (e << fmt.mbits) | m
    return (v << fmt.lsb) | rng.getrandbits(fmt.lsb)  # bits below the format: ignored


def cases_any(p, n: int, seed: int, aw: int, bw: int, cw: int):
    """As cases(), for any operand format: raw words, scaled and cancelling sums."""
    rng = random.Random(seed)
    L = p.L_max
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a = [rng.getrandbits(aw) for _ in range(L)]
            b = [rng.getrandbits(bw) for _ in range(L)]
            c = rng.getrandbits(cw)
        else:
            base = rng.randint(1, max(1, (1 << p.fa.ebits) - 2))
            a = [scaled_word(rng, p.fa, base - 2, base + 2) for _ in range(L)]
            b = [scaled_word(rng, p.fb, p.fb.bias - 2, p.fb.bias + 2) for _ in range(L)]
            e = base - p.fa.bias + p.fc.bias  # c near the products
            c = scaled_word(rng, p.fc, e - 3, e + 3)
            if kind == 2:
                dec = decode(p.fd, t_fdpa(p, a, b, 0))
                if dec.kind == "num" and dec.sig:
                    c = encode(p.fc, not dec.neg, dec.sig, dec.exp - p.fd.mbits, "rne")
        yield a, b, c


def cases(p, n: int, seed: int):
    rng = random.Random(seed)
    L = p.L_max
    c16 = p.fc is FP16
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a = [rng.getrandbits(16) for _ in range(L)]
            b = [rng.getrandbits(16) for _ in range(L)]
            c = rng.getrandbits(p.fc.width)
        else:
            base = rng.randint(3, 27)
            a = [f16_scaled(rng, base - 2, base + 2) for _ in range(L)]
            b = [f16_scaled(rng, 13, 17) for _ in range(L)]
            # products are ~2**(base - 15); put c in the same range
            if c16:
                c = f16_scaled(rng, max(base - 3, 1), min(base + 3, 30))
            else:
                c = f32_scaled(rng, base - 15 + 127 - 3, base - 15 + 127 + 3)
            if kind == 2:
                dec = decode(p.fd, t_fdpa(p, a, b, 0))
                if dec.kind == "num" and dec.sig:
                    # -sum in C's format: the result keeps only what alignment changes
                    c = encode(p.fc, not dec.neg, dec.sig, dec.exp - p.fd.mbits, "rne")
        yield a, b, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="f32.f16.f16.f32")
    ap.add_argument("--arch", default="Volta")
    ap.add_argument("--shape", default="")
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mutate", default="", help="model parameter overrides, e.g. F=24 (to test the testbench)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = base = tfdpa(args.arch, args.variant, args.shape)
    for kv in filter(None, args.mutate.split(",")):
        k, v = kv.split("=")
        p = dataclasses.replace(p, **{k: v if k == "rounding" else int(v)})
    tc = args.variant.split(".")[3]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        _, ta, tb, _ = args.variant.split(".")
        if base.fa is FP16 and base.fb is FP16:  # Volta's generator, so its vectors stay the same
            all_cases = list(cases(base, args.cases, args.seed))
            L = base.L_max
            all_cases += [(a + [0] * (L - len(a)), b + [0] * (L - len(b)), c_bits(c, tc))
                          for _, a, b, c in directed(tc)]
        else:
            aw, bw = PTX_TYPES[ta][1] * 8, PTX_TYPES[tb][1] * 8
            all_cases = list(cases_any(base, args.cases, args.seed, aw, bw, 32 if tc == "f32" else 16))
        for a, b, c in all_cases:
            d = t_fdpa(p, a, b, c)
            f.write(" ".join(f"{x:x}" for x in [*a, *b, c, d]) + "\n")
    print(f"{len(all_cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
