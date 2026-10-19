"""Cases for the Volta m8n8k4 harness (gpu/mma_sm70.cu) and the check of its output.

    python tools/gpu_cases.py gen   --variant f32.f16.f16.f32 --tiles 16384 --seed 1 --out DIR
    mma_sm70 f32.f16.f16.f32 <tiles> DIR/a.bin DIR/b.bin DIR/c.bin DIR/d.bin   (on the GPU host)
    python tools/gpu_cases.py check --dir DIR [--mmasim]

``gen`` writes random tiles (uniform raw bits) followed by the directed tiles of
``DIRECTED``, one case per tile in row 0, column 0, the rest zero. ``check`` compares every
element of ``d.bin`` with the integer model (and with MMA-Sim if asked) and writes
``check.json``. Needs only NumPy unless ``--mmasim``.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

from tensormirror.arch import tfdpa
from tensormirror.tfdpa import dot

WORD = {"f16": np.uint16, "f32": np.uint32}


def f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# name, a[4] (f16 bits), b[4] (f16 bits), c (a float, written in C's format); see H-002.
# Cases that only make sense with C in f32 are in F32_C_ONLY.
DIRECTED = [
    ("discrepancy_example", [f16(-(2.0**13)), f16(-0.5), f16(-0.25), f16(-0.125)],
     [f16(2.0**10), f16(1.0), f16(1.0), f16(1.0)], 2.0**23),
    ("e_zero_small_product", [f16((1 + 2.0**-10) * 2.0**-12), 0, 0, 0],
     [f16((1 + 3 * 2.0**-10) * 2.0**-12), 0, 0, 0], 0.0),
    ("negative_c_tiny_product", [f16(2.0**-15), 0, 0, 0], [f16(2.0**-15), 0, 0, 0], -1.0),
    ("positive_c_tiny_negative_product", [f16(-(2.0**-15)), 0, 0, 0], [f16(2.0**-15), 0, 0, 0], 1.0),
    ("exact_cancellation", [f16(1.0), f16(-1.0), 0, 0], [f16(1.0), f16(1.0), 0, 0], 0.0),
    ("minus_zero_c", [0, 0, 0, 0], [0, 0, 0, 0], -0.0),
    ("subnormal_c", [0, 0, 0, 0], [0, 0, 0, 0], "min_subnormal"),
    ("nan_payload", [0x7E01, 0, 0, 0], [f16(1.0), 0, 0, 0], 0.0),
    ("inf_minus_inf", [0x7C00, 0xFC00, 0, 0], [f16(1.0), f16(1.0), 0, 0], 0.0),
    ("inf_times_zero", [0x7C00, 0, 0, 0], [0, 0, 0, 0], 0.0),
]


F32_C_ONLY = {"discrepancy_example"}  # C = 2**23 does not fit in f16


def directed(tc: str) -> list:
    return [d for d in DIRECTED if tc == "f32" or d[0] not in F32_C_ONLY]


def c_bits(x, fmt: str) -> int:
    if x == "min_subnormal":
        return 1
    return f16(x) if fmt == "f16" else f32(x)


def gen(args) -> None:
    td, ta, tb, tc = args.variant.split(".")
    rng = np.random.default_rng(args.seed)
    n, nd = args.tiles, len(directed(tc))
    a = rng.integers(0, 1 << 16, size=(n + nd, 8, 4), dtype=np.uint64).astype(np.uint16)
    b = rng.integers(0, 1 << 16, size=(n + nd, 8, 4), dtype=np.uint64).astype(np.uint16)  # B^T: (n, k)
    cw = WORD[tc]
    c = rng.integers(0, 1 << (8 * np.dtype(cw).itemsize), size=(n + nd, 8, 8), dtype=np.uint64).astype(cw)
    cases = directed(tc)
    for i, (_, av, bv, cv) in enumerate(cases):
        t = n + i
        a[t], b[t], c[t] = 0, 0, 0
        a[t, 0], b[t, 0], c[t, 0, 0] = av, bv, c_bits(cv, tc)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("a", a), ("b", b), ("c", c)):
        x.tofile(out / f"{name}.bin")
    meta = {"variant": args.variant, "random_tiles": n, "directed": [d[0] for d in directed(tc)],
            "tiles": n + nd, "seed": args.seed}
    (out / "cases.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def check(args) -> int:
    d_dir = pathlib.Path(args.dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    td, ta, tb, tc = meta["variant"].split(".")
    nt = meta["tiles"]
    a = np.fromfile(d_dir / "a.bin", np.uint16).reshape(nt, 8, 4)
    b = np.fromfile(d_dir / "b.bin", np.uint16).reshape(nt, 8, 4)
    c = np.fromfile(d_dir / "c.bin", WORD[tc]).reshape(nt, 8, 8)
    d = np.fromfile(d_dir / "d.bin", WORD[td]).reshape(nt, 8, 8)
    p = tfdpa("Volta", meta["variant"])

    model = np.empty_like(d)
    for t in range(nt):
        for i in range(8):
            ai = a[t, i].tolist()
            for j in range(8):
                model[t, i, j] = dot(p, ai, b[t, j].tolist(), int(c[t, i, j]))
    report = {"variant": meta["variant"], "tiles": nt, "elements": int(d.size),
              "gpu_vs_model": int((d != model).sum())}
    if args.mmasim:
        import torch
        from mmasim.nv_ptx import MMA

        sim = MMA("Volta", f"m8n8k4.{meta['variant']}")
        tt = {"f16": (torch.float16, np.int16), "f32": (torch.float32, np.int32)}
        A = torch.from_numpy(np.repeat(a[:, :, None, :], 8, axis=2).view(np.int16)).view(torch.float16)
        B = torch.from_numpy(np.repeat(b[:, None, :, :], 8, axis=1).view(np.int16)).view(torch.float16)
        C = torch.from_numpy(c.view(tt[tc][1])).view(tt[tc][0])
        s = sim.dpa(A, B, C).view({"f16": torch.int16, "f32": torch.int32}[td]).numpy().view(WORD[td])
        report["gpu_vs_mmasim"] = int((d != s).sum())
        report["model_vs_mmasim"] = int((model != s).sum())
    directed = {}
    for k, name in enumerate(meta["directed"]):
        t = meta["random_tiles"] + k
        directed[name] = {"gpu": hex(d[t, 0, 0]), "model": hex(model[t, 0, 0]),
                          "rest_of_tile_ok": bool((d[t] == model[t]).all())}
    report["directed"] = directed
    bad = np.argwhere(d != model)[:10]
    report["first_mismatches"] = [
        {"tile": int(t), "i": int(i), "j": int(j), "a": [hex(x) for x in a[t, i]], "b": [hex(x) for x in b[t, j]],
         "c": hex(c[t, i, j]), "gpu": hex(d[t, i, j]), "model": hex(model[t, i, j])} for t, i, j in bad]
    (d_dir / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "first_mismatches"}, indent=1))
    return 1 if report["gpu_vs_model"] or report.get("gpu_vs_mmasim") else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--variant", default="f32.f16.f16.f32")
    g.add_argument("--tiles", type=int, default=16384)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--mmasim", action="store_true")
    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args)
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
