"""Bound the exponent a zero accumulator takes (e_zero) on a GPU, for an instruction with C
in f16 and A, B in f16.

Each tile holds one product a * b with C = +0: a = 2047 * 2**(ea - 10), b = 1025 *
2**(eb - 10), whose significand 2,098,175 ends in binary 1111, and ea + eb = e_p sweeps
-14 .. -28 (both normal). The zero C sits at e_zero; when e_zero > e_p + (F - 20) the
product loses that many low bits in the alignment, and an f32 result shows how many. The
analysis reruns the integer model with every e_zero from -14 to -45 and keeps those that
reproduce the GPU on every tile.

    python tools/probe_ezero.py gen <arch> <inst> <dir>     (inst with C in f16, e.g. m8n8k4.f32.f16.f16.f16)
    then run the harness on <dir> as for gpu_cases.py (gpu_batch.sh run <root> ...)
    python tools/probe_ezero.py analyze <dir>
"""

import dataclasses
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402


def f16_bits(sig11: int, e: int) -> int:
    """A normal f16 with significand sig11 (1024..2047) and unbiased exponent e."""
    return ((e + 15) << 10) | (sig11 - 1024)


def probes():
    out = []
    for e_p in range(-14, -29, -1):
        ea = max(e_p + 14, -14) if e_p < -14 else -7
        eb = e_p - ea
        out.append((e_p, f16_bits(2047, ea), f16_bits(1025, eb)))
    return out


def gen(arch, inst, d_dir):
    ins = gpu_cases.parse(inst)
    assert ins["ta"] == ins["tb"] == ins["tc"] == "f16", "A, B and C in f16"
    M, N, K = ins["M"], ins["N"], ins["K"]
    ps = probes()
    nt = len(ps)
    a = np.zeros((nt, M, K), np.uint16)
    b = np.zeros((nt, N, K), np.uint16)
    c = np.zeros((nt, M, N), np.uint16)
    for t, (_, x, y) in enumerate(ps):
        a[t, 0, 0], b[t, 0, 0] = x, y
    d_dir = pathlib.Path(d_dir)
    d_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("a", a), ("b", b), ("c", c)):
        arr.tofile(d_dir / f"{name}.bin")
    meta = {"arch": arch, "inst": inst, "tiles": nt, "random_tiles": nt, "scaled_from": nt, "directed": [],
            "seed": 0, "probe": [list(p) for p in ps]}
    (d_dir / "cases.json").write_text(json.dumps(meta) + "\n")


def analyze(d_dir):
    d_dir = pathlib.Path(d_dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    ins, a, b, c = gpu_cases.load(d_dir, meta)
    d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
    base = tfdpa(meta["arch"], ins["types"], ins["shape"])
    rows = []
    for t, (e_p, _a, _b) in enumerate(meta["probe"]):
        got = int(d[t, 0, 0])
        val = struct.unpack("<f", struct.pack("<I", got))[0] if ins["td"] == "f32" else None
        exact = 2047 * 1025 * 2.0 ** (e_p - 20)
        lost = None if val is None else round(np.log2(2047 * 1025 - val / 2.0 ** (e_p - 20) + 1))
        rows.append({"e_p": e_p, "gpu": hex(got), "value": val, "exact": exact, "lost_bits": lost,
                     "rest_zero": bool((d[t].ravel()[1:] == 0).all())})
    fits = []
    for ez in range(-14, -46, -1):
        p = dataclasses.replace(base, e_zero=ez)
        ok = all(dot(p, a[t, 0].tolist(), b[t, 0].tolist(), int(c[t, 0, 0])) == int(d[t, 0, 0])
                 for t in range(len(meta["probe"])))
        if ok:
            fits.append(ez)
    out = {"arch": meta["arch"], "inst": meta["inst"], "F": base.F, "e_zero_consistent": fits, "probes": rows}
    print(json.dumps(out, indent=1))
    (d_dir / "ezero.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        analyze(sys.argv[2])
