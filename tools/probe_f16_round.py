"""Find how far below a rounding tie a small product still counts, for an instruction with
A and B in f16 and C = +0, and which operand kinds reach the sum.

Each tile holds three products in row 0, column 0: a big one, a second one that puts the sum
exactly on a rounding tie of an f16 result, and a tiny one 2**-k below the tie bit, built
from normal * normal, normal * subnormal or subnormal * subnormal operands. With D in f16 a
kept tiny product rounds the tie up and a dropped one leaves it to go to even; with D in
f32 the sum shows directly. Two levels: a normal f16 result (1 + 2**-11 + tiny) and a
subnormal one (2**-25 + tiny, the tie of 2**-24). The analysis reruns the integer model
with every F from 10 to 40 and keeps those that reproduce the GPU on every tile.

    python tools/probe_f16_round.py gen <arch> <inst> <dir>
    then run the harness on <dir> as for gpu_cases.py (gpu_batch.sh run <root> ...)
    python tools/probe_f16_round.py analyze <dir>
"""

import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402


def pow2(e: int) -> int:
    """The f16 bits of 2**e, normal (-14..15) or subnormal (-24..-15)."""
    return (e + 15) << 10 if e >= -14 else 1 << (e + 24)


def split(e: int, kind: str):
    """Exponents (ea, eb) with ea + eb = e and the operand kinds asked, or None."""
    ranges = {"n": range(-14, 16), "s": range(-24, -14)}
    for ea in ranges[kind[0]]:
        if e - ea in ranges[kind[1]]:
            return ea, e - ea
    return None


LEVELS = {  # name: (big product, tie product), as (ea, eb) of powers of two
    "normal": ((0, 0), (-5, -6)),     # 1 + 2**-11: the tie between 1 and 1 + 2**-10
    "subnormal": ((-12, -13), None),  # 2**-25: the tie between 0 and 2**-24
}
TIE_BIT = {"normal": -11, "subnormal": -25}


def probes():
    out = []
    for level, (big, tie) in LEVELS.items():
        out.append({"level": level, "k": None, "kind": None, "terms": [big] + ([tie] if tie else [])})
        for k in range(1, 26):
            for kind in ("nn", "ns", "ss"):
                s = split(TIE_BIT[level] - k, kind)
                if s:
                    out.append({"level": level, "k": k, "kind": kind,
                                "terms": [big] + ([tie] if tie else []) + [s]})
    return out


def gen(arch, inst, d_dir):
    ins = gpu_cases.parse(inst)
    assert ins["ta"] == ins["tb"] == "f16", "A and B in f16"
    M, N, K = ins["M"], ins["N"], ins["K"]
    ps = probes()
    nt = len(ps)
    word = {2: np.uint16, 4: np.uint32}
    a = np.zeros((nt, M, K), np.uint16)
    b = np.zeros((nt, N, K), np.uint16)
    c = np.zeros((nt, M, N), word[ins["size"][ins["tc"]]])
    for t, p in enumerate(ps):
        for i, (ea, eb) in enumerate(p["terms"]):
            a[t, 0, i], b[t, 0, i] = pow2(ea), pow2(eb)
    d_dir = pathlib.Path(d_dir)
    d_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("a", a), ("b", b), ("c", c)):
        arr.tofile(d_dir / f"{name}.bin")
    meta = {"arch": arch, "inst": inst, "tiles": nt, "random_tiles": nt, "scaled_from": nt, "directed": [],
            "seed": 0, "probe": ps}
    (d_dir / "cases.json").write_text(json.dumps(meta) + "\n")


def analyze(d_dir):
    d_dir = pathlib.Path(d_dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    ins, a, b, c = gpu_cases.load(d_dir, meta)
    d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
    base = tfdpa(meta["arch"], ins["types"], ins["shape"])

    def model(p, t):
        return dot(p, a[t, 0].tolist(), b[t, 0].tolist(), int(c[t, 0, 0]))

    rows = [{"level": p["level"], "k": p["k"], "kind": p["kind"], "gpu": hex(int(d[t, 0, 0])),
             "model": hex(model(base, t)), "rest_zero": bool((d[t].ravel()[1:] == 0).all())}
            for t, p in enumerate(meta["probe"])]
    def fit(**kw):
        return all(model(dataclasses.replace(base, **kw), t) == int(d[t, 0, 0]) for t in range(len(rows)))

    out = {"arch": meta["arch"], "inst": meta["inst"], "F": base.F, "e_zero": base.e_zero,
           "mismatches": sum(r["gpu"] != r["model"] for r in rows),
           "F_consistent": [F for F in range(10, 41) if fit(F=F)],
           "e_zero_consistent": [ez for ez in range(-14, -46, -1) if fit(e_zero=ez)], "probes": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "probes"}))
    for r in rows:
        if r["gpu"] != r["model"]:
            print(json.dumps(r))
    (d_dir / "round.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        analyze(sys.argv[2])
