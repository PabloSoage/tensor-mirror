"""Test vectors for rtl/tfdpa_f16_f32.sv from the integer model.

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

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.formats import FP32, decode  # noqa: E402
from tensormirror.tfdpa import t_fdpa  # noqa: E402


def f16_scaled(rng: random.Random, lo: int, hi: int) -> int:
    e = rng.randint(lo, hi)  # biased exponent
    return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)


def f32_scaled(rng: random.Random, lo: int, hi: int) -> int:
    return (rng.getrandbits(1) << 31) | (rng.randint(lo, hi) << 23) | rng.getrandbits(23)


def cases(p, n: int, seed: int):
    rng = random.Random(seed)
    L = p.L_max
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a = [rng.getrandbits(16) for _ in range(L)]
            b = [rng.getrandbits(16) for _ in range(L)]
            c = rng.getrandbits(32)
        else:
            base = rng.randint(3, 27)
            a = [f16_scaled(rng, base - 2, base + 2) for _ in range(L)]
            b = [f16_scaled(rng, 13, 17) for _ in range(L)]
            # products are ~2**(base - 15); put c in the same range (f32 bias 127)
            c = f32_scaled(rng, base - 15 + 127 - 3, base - 15 + 127 + 3)
            if kind == 2:
                s = t_fdpa(p, a, b, 0)
                dec = decode(FP32, s)
                if dec.kind == "num" and dec.sig:
                    c = s ^ (1 << 31)  # -sum: the result keeps only what alignment changes
        yield a, b, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="f32.f16.f16.f32")
    ap.add_argument("--arch", default="Volta")
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mutate", default="", help="model parameter overrides, e.g. F=24 (to test the testbench)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = tfdpa(args.arch, args.variant)
    for kv in filter(None, args.mutate.split(",")):
        k, v = kv.split("=")
        p = dataclasses.replace(p, **{k: v if k == "rounding" else int(v)})
    tc = args.variant.split(".")[3]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        all_cases = list(cases(tfdpa(args.arch, args.variant), args.cases, args.seed))
        all_cases += [(a, b, c_bits(c, tc)) for _, a, b, c in directed(tc)]
        for a, b, c in all_cases:
            d = t_fdpa(p, a, b, c)
            f.write(" ".join(f"{x:x}" for x in [*a, *b, c, d]) + "\n")
    print(f"{len(all_cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
