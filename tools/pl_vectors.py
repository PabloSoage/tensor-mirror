"""Vectors for rtl/tm_pl.v (its testbench and the board): one case per line, hex,
a[0..3] b[0..3] c d_f32 d_f32c16 d_f16, where the three d are Volta's three m8n8k4
variants on the same a, b and c (the f16-C units read c[15:0]).

The cases come from tools/rtl_vectors.py's generator for one variant (--source), so that
variant gets raw, scaled and cancelling cases and the directed ones; the other two units
see the same words and are checked too. Run it once per source for full coverage.

    python tools/pl_vectors.py --source f32.f16.f16.f16 --cases 100000 --seed 1 --out v.txt
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gpu_cases import c_bits, directed  # noqa: E402
from rtl_vectors import cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.tfdpa import t_fdpa  # noqa: E402

VARIANTS = ("f32.f16.f16.f32", "f32.f16.f16.f16", "f16.f16.f16.f16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="f32.f16.f16.f32", choices=VARIANTS)
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ps = [tfdpa("Volta", v) for v in VARIANTS]
    src = tfdpa("Volta", args.source)
    tc = args.source.split(".")[3]
    rows = list(cases(src, args.cases, args.seed))
    rows += [(a + [0] * (4 - len(a)), b + [0] * (4 - len(b)), c_bits(c, tc)) for _, a, b, c in directed(tc)]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for a, b, c in rows:
            ds = [t_fdpa(p, a, b, c if p.fc.width == 32 else c & 0xFFFF) for p in ps]
            f.write(" ".join(f"{x:x}" for x in [*a, *b, c, *ds]) + "\n")
    print(f"{len(rows)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
