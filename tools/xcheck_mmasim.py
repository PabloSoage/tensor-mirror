"""Compare tensor-mirror's integer T-FDPA with MMA-Sim on random raw bit patterns.

Every input word is uniformly random, so NaNs, infinities, subnormals and zeros all
appear, as in MMA-Sim's own GPU tests. Each case is one dot product of length K (chained
in groups of L_max where K is larger). Prints one JSON line and the first mismatches;
exits non-zero if any bit differs.

    python tools/xcheck_mmasim.py --arch Volta --inst m8n8k4.f32.f16.f16.f32 --cases 1000000
    python tools/xcheck_mmasim.py --all --cases 100000   # every T-FDPA mma.sync MMA-Sim has
"""

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import torch
from mmasim.nv_ptx import MMA
from mmasim.nv_ptx.isa_mma import arch_mma_shape_and_type

from tensormirror.arch import tfdpa
from tensormirror.tfdpa import dot

# PTX type -> (torch dtype MMA-Sim takes, bits in the register word)
TORCH = {"f32": (torch.float32, 32), "tf32": (torch.float32, 32), "f16": (torch.float16, 16),
         "bf16": (torch.bfloat16, 16), "e5m2": (torch.float8_e5m2, 8), "e4m3": (torch.float8_e4m3fn, 8),
         "e3m2": (torch.uint8, 8), "e2m3": (torch.uint8, 8), "e2m1": (torch.uint8, 8)}
UNSIGNED = {32: np.uint32, 16: np.uint16, 8: np.uint8}
SIGNED = {32: np.int32, 16: np.int16, 8: np.int8}


def raw(rng: np.random.Generator, shape: tuple[int, ...], t: str) -> np.ndarray:
    bits = TORCH[t][1]
    return rng.integers(0, 1 << bits, size=shape, dtype=np.uint64).astype(UNSIGNED[bits])


def as_torch(x: np.ndarray, t: str) -> torch.Tensor:
    dtype, bits = TORCH[t]
    if dtype == torch.uint8:
        return torch.from_numpy(x.copy())
    return torch.from_numpy(x.view(SIGNED[bits])).view(dtype)  # torch has no uint16/32 views


def check(arch: str, inst: str, cases: int, seed: int, mutate: str = "", show: int = 10,
          measured: bool = False) -> dict:
    shape, types = inst.split(".", 1)
    td, ta, tb, tc = types.split(".")
    k = int(shape.split("k")[1])
    p = tfdpa(arch, types, shape, measured=measured)  # published (False) compares with MMA-Sim
    for kv in filter(None, mutate.split(",")):  # mutants: the check must catch them
        key, v = kv.split("=")
        p = dataclasses.replace(p, **{key: v if key == "rounding" else int(v)})
    sim = MMA(arch, inst)
    rng = np.random.default_rng(seed)
    a, b, c = raw(rng, (cases, k), ta), raw(rng, (cases, k), tb), raw(rng, (cases,), tc)

    t0 = time.time()
    d_sim = sim.dpa(as_torch(a, ta), as_torch(b, tb), as_torch(c, tc))
    bits = TORCH[td][1]
    d_sim = d_sim.view({32: torch.int32, 16: torch.int16}[bits]).numpy().view(UNSIGNED[bits])
    t_sim = time.time() - t0

    t0 = time.time()
    bad = [i for i in range(cases) if dot(p, a[i].tolist(), b[i].tolist(), int(c[i])) != int(d_sim[i])]
    t_ref = time.time() - t0
    for i in bad[:show]:
        print(f"  case {i}: a={[hex(x) for x in a[i]]} b={[hex(x) for x in b[i]]} c={hex(c[i])} "
              f"mmasim={hex(d_sim[i])} model={hex(dot(p, a[i].tolist(), b[i].tolist(), int(c[i])))}",
              file=sys.stderr)
    return {"arch": arch, "inst": inst, "L": p.L_max, "F": p.F, "cases": cases, "seed": seed, "mutate": mutate,
            "measured": measured, "mismatches": len(bad), "t_mmasim_s": round(t_sim, 1), "t_model_s": round(t_ref, 1)}


def tfdpa_instructions():
    """(arch, instruction) for every non-FP64 mma.sync MMA-Sim models (all are T-FDPA)."""
    for arch, insts in arch_mma_shape_and_type.items():
        for inst in insts:
            if ".f64." not in inst:
                yield arch, inst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Volta")
    ap.add_argument("--inst", default="m8n8k4.f32.f16.f16.f32", help="shape.D.A.B.C as in PTX")
    ap.add_argument("--all", action="store_true", help="every T-FDPA mma.sync instruction")
    ap.add_argument("--cases", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--measured", action="store_true", help="the model with what we measured on GPUs")
    ap.add_argument("--mutate", default="", help="override the integer model's parameters, e.g. F=24,rounding=rne")
    args = ap.parse_args()

    jobs = list(tfdpa_instructions()) if args.all else [(args.arch, args.inst)]
    failed = 0
    for arch, inst in jobs:
        r = check(arch, inst, args.cases, args.seed, args.mutate, measured=args.measured)
        print(json.dumps(r), flush=True)
        failed += r["mismatches"] > 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
