"""Search T-FDPA parameters (L, F, rounding, e_zero) that reproduce a GPU's output.

For an instruction whose GPU output disagrees with the published parameters: every
combination on the grid is run on the measured cases and ranked by mismatches. A
combination with 0 mismatches over all cases (raw and scaled tiles) is a candidate model.
L < K means groups of L chained through D's format, as on Ampere and later.

    python tools/fit_tfdpa.py <case dir> [--tiles N] [--F 22,23,24,25] [--L 1,2,4]
"""

import argparse
import dataclasses
import itertools
import json
import multiprocessing
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402


def mismatches(job):
    p, a, b, c, d = job
    bad = 0
    for t in range(c.shape[0]):
        for i in range(c.shape[1]):
            ai = a[t, i].tolist()
            for j in range(c.shape[2]):
                bad += dot(p, ai, b[t, j].tolist(), int(c[t, i, j])) != int(d[t, i, j])
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tiles", type=int, default=256, help="tiles from each half (raw, scaled)")
    ap.add_argument("--F", default="22,23,24,25")
    ap.add_argument("--L", default="1,2,4")
    ap.add_argument("--e-zero", default="")
    ap.add_argument("--rounding", default="", help="e.g. rz,rne (default: the model's)")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    d_dir = pathlib.Path(args.dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    ins, a, b, c = gpu_cases.load(d_dir, meta)
    d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
    half, n = meta["scaled_from"], meta["random_tiles"]
    ix = np.r_[0 : min(args.tiles, half), half : min(half + args.tiles, n)]
    a, b, c, d = a[ix], b[ix], c[ix], d[ix]
    base = tfdpa(meta["arch"], ins["types"], ins["shape"])
    ez = [int(x) for x in args.e_zero.split(",")] if args.e_zero else [base.e_zero]
    rnds = args.rounding.split(",") if args.rounding else [base.rounding]
    grid = [dataclasses.replace(base, L_max=L, F=F, e_zero=e, rounding=r)
            for L, F, e, r in itertools.product(map(int, args.L.split(",")), map(int, args.F.split(",")), ez, rnds)]
    with multiprocessing.Pool(args.jobs) as pool:
        counts = pool.map(mismatches, [(p, a, b, c, d) for p in grid])
    print(f"{meta['inst']} on {meta['arch']} model; {c.size} elements from {len(ix)} tiles")
    for p, k in sorted(zip(grid, counts, strict=True), key=lambda x: x[1]):
        print(json.dumps({"L": p.L_max, "F": p.F, "e_zero": p.e_zero, "rounding": p.rounding, "mismatches": k}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
