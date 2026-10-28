"""Cases for the mma.sync harnesses (gpu/mma_sm70.cu, gpu/mma_sync.cu) and their check.

    python tools/gpu_cases.py gen   --arch Turing --inst m16n8k8.f32.f16.f16.f32 --tiles 4096 --out DIR
    <harness> <inst> <tiles> DIR/a.bin DIR/b.bin DIR/c.bin DIR/d.bin       (on the GPU host)
    python tools/gpu_cases.py check --dir DIR [--mmasim] [--jobs N]

Instruction names are shape.D.A.B.C as in MMA-Sim, with ``f8f6f4.`` after the shape for
RTX Blackwell's ``kind::f8f6f4`` (e.g. ``m16n8k32.f8f6f4.f32.e2m1.e2m1.f32``).

``gen`` writes, per tile, M x K of A, N x K of B^T and M x N of C, then D comes back M x N.
With ``--steps S`` A and B are M x S*K and N x S*K, for gpu/mma_sync.cu's loop of S
instructions (pass S as its last argument); the model chains the whole row.
Half the random tiles are uniform raw bits (NaN, infinities and subnormals appear); the
other half are "scaled": every operand has an exponent within a few binades of 1, so the
products overlap and every fractional bit of the fused sum shows. Bits a format does not
use (the low 13 of a TF32 word, bits 7:6 and 1:0 around FP4, bits 7:6 above FP6) are
random too, so the check also tests that the GPU ignores them. The directed cases of
``DIRECTED`` (f16 operands only) follow, one per tile in row 0, column 0, rest zero.

``check`` compares every element of ``d.bin`` with the integer model (and with MMA-Sim if
asked and if it models the instruction) and writes ``check.json``.
"""

import argparse
import json
import multiprocessing
import pathlib
import random
import struct
import sys

import numpy as np

from tensormirror.arch import PTX_TYPES, tfdpa
from tensormirror.formats import FN, IEEE, Format
from tensormirror.tfdpa import dot

WORD = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# name, a (f16 bits), b (f16 bits), c (a float, written in C's format); see H-002. Padded
# with zeros to K. Cases that only make sense with C in f32 are in F32_C_ONLY.
DIRECTED = [
    ("discrepancy_example", [f16(-(2.0**13)), f16(-0.5), f16(-0.25), f16(-0.125)],
     [f16(2.0**10), f16(1.0), f16(1.0), f16(1.0)], 2.0**23),
    ("e_zero_small_product", [f16((1 + 2.0**-10) * 2.0**-12)], [f16((1 + 3 * 2.0**-10) * 2.0**-12)], 0.0),
    # 2**-25 + 2**-43 with C = +0: in f16 it rounds up to 2**-24 unless e_zero drops the
    # 2**-43, which leaves an exact tie that goes to 0 (the subnormal 2**-24 times 2**-19)
    ("e_zero_rne_tie", [f16(2.0**-12), 0x0001], [f16(2.0**-13), f16(2.0**-19)], 0.0),
    ("negative_c_tiny_product", [f16(2.0**-15)], [f16(2.0**-15)], -1.0),
    ("positive_c_tiny_negative_product", [f16(-(2.0**-15))], [f16(2.0**-15)], 1.0),
    ("exact_cancellation", [f16(1.0), f16(-1.0)], [f16(1.0), f16(1.0)], 0.0),
    ("minus_zero_c", [0], [0], -0.0),
    ("subnormal_c", [0], [0], "min_subnormal"),
    ("nan_payload", [0x7E01], [f16(1.0)], 0.0),
    ("inf_minus_inf", [0x7C00, 0xFC00], [f16(1.0), f16(1.0)], 0.0),
    ("inf_times_zero", [0x7C00], [0], 0.0),
]
F32_C_ONLY = {"discrepancy_example"}  # C = 2**23 does not fit in f16


def directed(tc: str) -> list:
    return [d for d in DIRECTED if tc == "f32" or d[0] not in F32_C_ONLY]


def c_bits(x, fmt: str) -> int:
    if x == "min_subnormal":
        return 1
    return f16(x) if fmt == "f16" else f32(x)


def parse(inst: str) -> dict:
    """Shape, PTX types and register sizes of an instruction name."""
    shape, rest = inst.split(".", 1)
    kind = ""
    if rest.startswith("f8f6f4."):
        kind, rest = "f8f6f4", rest[len("f8f6f4."):]
    td, ta, tb, tc = rest.split(".")
    m, nk = shape[1:].split("n")
    n, k = nk.split("k")
    return {"shape": shape, "kind": kind, "types": rest, "td": td, "ta": ta, "tb": tb, "tc": tc,
            "M": int(m), "N": int(n), "K": int(k), "size": {t: PTX_TYPES[t][1] for t in (td, ta, tb, tc)}}


def scaled_word(rng: random.Random, fmt: Format, size: int, spread: int) -> int:
    """A word of ``size`` bytes holding a value of ``fmt`` within ``spread`` binades of 1."""
    top = (1 << fmt.ebits) - (2 if fmt.specials == IEEE else 1)
    e = min(max(fmt.bias + rng.randint(-spread, spread), 1), top)
    frac = rng.getrandbits(fmt.mbits)
    if fmt.specials == FN and e == top:
        frac &= (1 << fmt.mbits) - 2  # not the NaN pattern
    field = (rng.getrandbits(1) << (fmt.width - 1)) | (e << fmt.mbits) | frac
    unused = rng.getrandbits(8 * size) & ~(((1 << fmt.width) - 1) << fmt.lsb)  # bits the format ignores
    return (field << fmt.lsb) | unused


def gen(args) -> None:
    ins = parse(args.inst)
    M, N, K, size = ins["M"], ins["N"], ins["K"] * getattr(args, "steps", 1), ins["size"]
    cases = directed(ins["tc"]) if ins["ta"] == ins["tb"] == "f16" and K >= 4 else []
    n, nd = args.tiles, len(cases)
    nprng = np.random.default_rng(args.seed)

    def raw(shape, t):
        s = size[t]
        return nprng.integers(0, 1 << (8 * s), size=shape, dtype=np.uint64).astype(WORD[s])

    a, b, c = raw((n + nd, M, K), ins["ta"]), raw((n + nd, N, K), ins["tb"]), raw((n + nd, M, N), ins["tc"])
    rng = random.Random(args.seed)
    fa, fb, fc = PTX_TYPES[ins["ta"]][0], PTX_TYPES[ins["tb"]][0], PTX_TYPES[ins["tc"]][0]
    for t in range(n // 2, n):  # the scaled half
        a[t] = [[scaled_word(rng, fa, size[ins["ta"]], 2) for _ in range(K)] for _ in range(M)]
        b[t] = [[scaled_word(rng, fb, size[ins["tb"]], 2) for _ in range(K)] for _ in range(N)]
        c[t] = [[scaled_word(rng, fc, size[ins["tc"]], 3) for _ in range(N)] for _ in range(M)]
    for i, (_, av, bv, cv) in enumerate(cases):
        t = n + i
        a[t], b[t], c[t] = 0, 0, 0
        a[t, 0, : len(av)], b[t, 0, : len(bv)], c[t, 0, 0] = av, bv, c_bits(cv, ins["tc"])
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("a", a), ("b", b), ("c", c)):
        x.tofile(out / f"{name}.bin")
    meta = {"arch": args.arch, "inst": args.inst, "random_tiles": n, "scaled_from": n // 2,
            "directed": [d[0] for d in cases], "tiles": n + nd, "seed": args.seed,
            "steps": getattr(args, "steps", 1)}
    (out / "cases.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def load(d_dir: pathlib.Path, meta: dict):
    ins = parse(meta["inst"])
    M, N, K, size, nt = ins["M"], ins["N"], ins["K"] * meta.get("steps", 1), ins["size"], meta["tiles"]
    a = np.fromfile(d_dir / "a.bin", WORD[size[ins["ta"]]]).reshape(nt, M, K)
    b = np.fromfile(d_dir / "b.bin", WORD[size[ins["tb"]]]).reshape(nt, N, K)
    c = np.fromfile(d_dir / "c.bin", WORD[size[ins["tc"]]]).reshape(nt, M, N)
    return ins, a, b, c


def _model_tiles(job):
    arch, types, shape, K, a, b, c = job
    p = tfdpa(arch, types, shape)
    nt, M, N = c.shape
    out = np.empty((nt, M, N), dtype=np.uint64)
    for t in range(nt):
        for i in range(M):
            ai = a[t, i].tolist()
            for j in range(N):
                bj, acc = b[t, j].tolist(), int(c[t, i, j])
                for k0 in range(0, len(ai), K):  # one instruction after another (--steps)
                    acc = dot(p, ai[k0 : k0 + K], bj[k0 : k0 + K], acc)
                out[t, i, j] = acc
    return out


def model_output(arch: str, ins: dict, a, b, c, jobs: int) -> np.ndarray:
    chunks = np.array_split(np.arange(c.shape[0]), max(1, jobs * 4))
    work = [(arch, ins["types"], ins["shape"], ins["K"], a[ix], b[ix], c[ix]) for ix in chunks if len(ix)]
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            parts = pool.map(_model_tiles, work)
    else:
        parts = [_model_tiles(w) for w in work]
    return np.concatenate(parts).astype(WORD[ins["size"][ins["td"]]])


def mmasim_output(arch: str, ins: dict, a, b, c):
    """MMA-Sim's D, or None if it does not model the instruction."""
    import torch
    from mmasim.nv_ptx import MMA
    from mmasim.nv_ptx.isa_mma import arch_mma_shape_and_type

    name = f"{ins['shape']}.{ins['types']}"
    if name not in arch_mma_shape_and_type.get(arch, []):
        return None
    sim = MMA(arch, name)
    tt = {"f32": torch.float32, "tf32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16,
          "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    signed = {1: np.int8, 2: np.int16, 4: np.int32}

    def as_torch(x, t):
        if t not in tt:  # fp6 and fp4 go in as raw bytes
            return torch.from_numpy(x.copy())
        return torch.from_numpy(x.view(signed[x.dtype.itemsize])).view(tt[t])

    M, N = c.shape[1:]
    A = as_torch(np.repeat(a[:, :, None, :], N, axis=2), ins["ta"])
    B = as_torch(np.repeat(b[:, None, :, :], M, axis=1), ins["tb"])
    C = as_torch(c, ins["tc"])
    s = sim.dpa(A, B, C)
    bits = {2: torch.int16, 4: torch.int32}[s.element_size()]
    return s.view(bits).numpy().view(WORD[s.element_size()])


def check(args) -> int:
    d_dir = pathlib.Path(args.dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    ins, a, b, c = load(d_dir, meta)
    d = np.fromfile(d_dir / "d.bin", WORD[ins["size"][ins["td"]]]).reshape(c.shape)
    model = model_output(meta["arch"], ins, a, b, c, args.jobs)
    half = meta["scaled_from"]
    rnd = slice(0, meta["random_tiles"])
    report = {"arch": meta["arch"], "inst": meta["inst"], "tiles": meta["tiles"], "elements": int(d.size),
              "gpu_vs_model": int((d != model).sum()),
              "gpu_vs_model_raw_tiles": int((d[:half] != model[:half]).sum()),
              "gpu_vs_model_scaled_tiles": int((d[half:rnd.stop] != model[half:rnd.stop]).sum())}
    if args.mmasim and meta.get("steps", 1) == 1:  # MMA-Sim runs one instruction
        s = mmasim_output(meta["arch"], ins, a, b, c)
        report["gpu_vs_mmasim"] = None if s is None else int((d != s).sum())
        report["model_vs_mmasim"] = None if s is None else int((model != s).sum())
    report["directed"] = {
        name: {"gpu": hex(d[meta["random_tiles"] + k, 0, 0]), "model": hex(model[meta["random_tiles"] + k, 0, 0]),
               "rest_of_tile_ok": bool((d[meta["random_tiles"] + k] == model[meta["random_tiles"] + k]).all())}
        for k, name in enumerate(meta["directed"])}
    bad = np.argwhere(d != model)[:10]
    report["first_mismatches"] = [
        {"tile": int(t), "i": int(i), "j": int(j), "a": [hex(x) for x in a[t, i]], "b": [hex(x) for x in b[t, j]],
         "c": hex(c[t, i, j]), "gpu": hex(d[t, i, j]), "model": hex(model[t, i, j])} for t, i, j in bad]
    (d_dir / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("first_mismatches", "directed")}))
    return 1 if report["gpu_vs_model"] or report.get("gpu_vs_mmasim") else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--arch", default="Volta")
    g.add_argument("--inst", default="m8n8k4.f32.f16.f16.f32")
    g.add_argument("--tiles", type=int, default=4096)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--steps", type=int, default=1, help="instructions per tile, K after K")
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--mmasim", action="store_true")
    c.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args)
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
