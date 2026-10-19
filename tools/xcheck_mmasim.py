"""Compare tcmirror's integer T-FDPA with MMA-Sim on random raw bit patterns.

Every input word is uniformly random, so NaNs, infinities, subnormals and zeros all
appear, as in MMA-Sim's own GPU tests. Prints one JSON line per instruction and the
first mismatches; exits non-zero if any bit differs.

    python tools/xcheck_mmasim.py --arch Volta --types f32.f16.f16.f32 --cases 1000000
"""

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import torch
from mmasim.nv_ptx import MMA

from tcmirror.arch import tfdpa
from tcmirror.tfdpa import t_fdpa

TORCH = {"f16": (torch.float16, np.uint16), "bf16": (torch.bfloat16, np.uint16), "f32": (torch.float32, np.uint32)}


def raw(rng: np.random.Generator, shape: tuple[int, ...], t: str) -> np.ndarray:
    dt = TORCH[t][1]
    return rng.integers(0, np.iinfo(dt).max + 1, size=shape, dtype=np.uint64).astype(dt)


def as_torch(x: np.ndarray, t: str) -> torch.Tensor:
    signed = {np.uint16: np.int16, np.uint32: np.int32}[x.dtype.type]  # torch has no uint16 views
    return torch.from_numpy(x.view(signed)).view(TORCH[t][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Volta")
    ap.add_argument("--types", default="f32.f16.f16.f32", help="D.A.B.C as in PTX")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mutate", default="", help="override tcmirror's parameters, e.g. F=24,rounding=rne")
    args = ap.parse_args()

    td, ta, tb, tc = args.types.split(".")
    p = tfdpa(args.arch, args.types)
    for kv in filter(None, args.mutate.split(",")):  # mutants: the check must catch them
        k, v = kv.split("=")
        p = dataclasses.replace(p, **{k: v if k == "rounding" else int(v)})
    sim = MMA(args.arch, f"m8n8k{args.k}.{args.types}")
    rng = np.random.default_rng(args.seed)
    n = args.cases
    a, b, c = raw(rng, (n, args.k), ta), raw(rng, (n, args.k), tb), raw(rng, (n,), tc)

    t0 = time.time()
    d_sim = sim.dpa(as_torch(a, ta), as_torch(b, tb), as_torch(c, tc))
    d_sim = d_sim.view({2: torch.int16, 4: torch.int32}[d_sim.element_size()]).numpy().view(TORCH[td][1])
    t_sim = time.time() - t0

    t0 = time.time()
    bad = []
    for i in range(n):
        d = t_fdpa(p, a[i].tolist(), b[i].tolist(), int(c[i]))
        if d != int(d_sim[i]):
            bad.append(i)
    t_ref = time.time() - t0

    print(json.dumps({"arch": args.arch, "types": args.types, "k": args.k, "cases": n, "seed": args.seed,
                      "mutate": args.mutate,
                      "mismatches": len(bad), "t_mmasim_s": round(t_sim, 1), "t_tcmirror_s": round(t_ref, 1)}))
    for i in bad[:10]:
        print(f"  case {i}: a={[hex(x) for x in a[i]]} b={[hex(x) for x in b[i]]} c={hex(c[i])} "
              f"mmasim={hex(d_sim[i])} tcmirror={hex(t_fdpa(p, a[i].tolist(), b[i].tolist(), int(c[i])))}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
