"""Block-scale GEMM jobs for rtl/tm_pl_array.v with GST = 1, and their reference: a loop of
block-scale mma.sync per 16 x 8 tile (gpu/mma_block_scale.cu with steps, H-037).

    python tools/bs_gemm.py gen    --inst m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0 --jobs 96 --out DIR
    python tools/bs_gemm.py tiles  --dir DIR                  (DIR/tiles, tools/bs_cases.py's files)
    mma_block_scale <inst> <tiles> DIR/tiles/{a,b,c,sa,sb,sel,d}.bin <steps>      (on the GPU host)
    python tools/bs_cases.py check --dir DIR/tiles --steps <steps> --jobs N
    python tools/bs_gemm.py gather --dir DIR                  (DIR/tiles/gemm_d.bin)
    python tools/bs_gemm.py rtl    --dir DIR --R 4 --C 4      (DIR/in.bin, DIR/expected.bin)

A job is D = C + A B over K, one instruction of k elements after another, with A M x K and
B^T N x K one E2M1 element per byte (bits 3:0, as the harness takes them; bits 7:4 random),
C M x N f32, and the scale factors SF_A M x K/block and SF_B N x K/block, one per block of
the instruction's K / NB elements. The thirds of the jobs are those of ``bs_cases.py gen``:
everything raw, raw elements with scales near 1, then elements, C and scales near 1.

``tiles`` cuts every job into 16 x 8 output tiles, each a chain of K / k input tiles in
order (C from the first), with the scales placed in the lanes' registers by
``bs_cases.scale_regs`` under random valid selectors. ``rtl`` writes the beats of
rtl/tm_pl_array.v: per job A, B^T (two elements per byte, low nibble first) and C, then
SF_A as one beat per group of R rows and step, scale j of row i * R + r in byte r * NB + j,
then SF_B the same per group of C columns; and the expected D beats from the GPU.
"""

import argparse
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bs_cases import near_one, near_one_c, near_one_scale, parse, scale_regs, selectors, write  # noqa: E402

TM, TN = 16, 8


def gen(args) -> None:
    ins = parse(args.inst)
    k, nb = ins["K"], ins["blocks"]
    J, M, N, K = args.jobs, args.M, args.N, args.K
    assert M % TM == 0 and N % TN == 0 and K % k == 0 and J % 3 == 0
    nbk = K // k * nb  # scale factors per row over the whole of K
    nprng = np.random.default_rng(args.seed)
    rng = random.Random(args.seed)
    a = nprng.integers(0, 256, (J, M, K), dtype=np.uint8)
    b = nprng.integers(0, 256, (J, N, K), dtype=np.uint8)
    c = nprng.integers(0, 1 << 32, (J, M, N), dtype=np.uint64).astype(np.uint32)
    sfa = nprng.integers(0, 256, (J, M, nbk), dtype=np.uint8)
    sfb = nprng.integers(0, 256, (J, N, nbk), dtype=np.uint8)
    third = J // 3
    for j in range(third, J):  # scales near 1
        sfa[j] = [[near_one_scale(rng, ins["stype"]) for _ in range(nbk)] for _ in range(M)]
        sfb[j] = [[near_one_scale(rng, ins["stype"]) for _ in range(nbk)] for _ in range(N)]
    for j in range(2 * third, J):  # and elements and C near 1
        a[j] = [[near_one(rng, ins, ins["ta"]) for _ in range(K)] for _ in range(M)]
        b[j] = [[near_one(rng, ins, ins["tb"]) for _ in range(K)] for _ in range(N)]
        c[j] = [[near_one_c(rng) for _ in range(N)] for _ in range(M)]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("a", a), ("b", b), ("c", c), ("sfa", sfa), ("sfb", sfb)):
        x.tofile(out / f"gemm_{name}.bin")
    meta = {"inst": args.inst, "jobs": J, "M": M, "N": N, "K": K, "k": k, "blocks": nb,
            "thirds": third, "seed": args.seed}
    (out / "gemm.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def load(d: pathlib.Path):
    meta = json.loads((d / "gemm.json").read_text())
    J, M, N, K = meta["jobs"], meta["M"], meta["N"], meta["K"]
    nbk = K // meta["k"] * meta["blocks"]
    x = {"a": np.fromfile(d / "gemm_a.bin", np.uint8).reshape(J, M, K),
         "b": np.fromfile(d / "gemm_b.bin", np.uint8).reshape(J, N, K),
         "c": np.fromfile(d / "gemm_c.bin", np.uint32).reshape(J, M, N),
         "sfa": np.fromfile(d / "gemm_sfa.bin", np.uint8).reshape(J, M, nbk),
         "sfb": np.fromfile(d / "gemm_sfb.bin", np.uint8).reshape(J, N, nbk)}
    return meta, x


def tiles(args) -> None:
    d = pathlib.Path(args.dir)
    meta, x = load(d)
    J, M, N, K, k, nb = (meta[n] for n in ("jobs", "M", "N", "K", "k", "blocks"))
    S, mt, nt_ = K // k, M // TM, N // TN
    nout = J * mt * nt_
    ni = nout * S  # input tile t * S + s: output tile t = (job, tm, tn), step s
    rng = random.Random(meta["seed"] + 1)
    nprng = np.random.default_rng(meta["seed"] + 1)
    # (J, tm, tn, s, rows, k): tile rows and the step's slice of K
    a = x["a"].reshape(J, mt, 1, TM, S, k).transpose(0, 1, 2, 4, 3, 5)
    a = np.broadcast_to(a, (J, mt, nt_, S, TM, k)).reshape(ni, TM, k)
    b = x["b"].reshape(J, 1, nt_, TN, S, k).transpose(0, 1, 2, 4, 3, 5)
    b = np.broadcast_to(b, (J, mt, nt_, S, TN, k)).reshape(ni, TN, k)
    c = np.zeros((J, mt, nt_, S, TM, TN), np.uint32)
    c[:, :, :, 0] = x["c"].reshape(J, mt, TM, nt_, TN).transpose(0, 1, 3, 2, 4)
    c = c.reshape(ni, TM, TN)
    sfa = x["sfa"].reshape(J, mt, 1, TM, S, nb).transpose(0, 1, 2, 4, 3, 5)
    sfa = np.broadcast_to(sfa, (J, mt, nt_, S, TM, nb)).reshape(ni, TM, nb)
    sfb = x["sfb"].reshape(J, 1, nt_, TN, S, nb).transpose(0, 1, 2, 4, 5, 3)
    sfb = np.broadcast_to(sfb, (J, mt, nt_, S, nb, TN)).reshape(ni, nb, TN)
    sel = np.array([selectors(rng, nb) for _ in range(ni)], dtype=np.uint16)
    sa = np.empty((ni, 32), np.uint32)
    sb = np.empty((ni, 32), np.uint32)
    for t in range(ni):
        sa[t], sb[t] = scale_regs(sfa[t], sfb[t], sel[t], nprng)
    write(d / "tiles", a=np.ascontiguousarray(a), b=np.ascontiguousarray(b), c=c,
          sfa=np.ascontiguousarray(sfa), sfb=np.ascontiguousarray(sfb), sa=sa, sb=sb, sel=sel,
          meta={"inst": meta["inst"], "tiles": ni, "random_tiles": ni, "thirds": meta["thirds"],
                "directed": [], "seed": meta["seed"], "steps": S, "output_tiles": nout})


def gather(args) -> None:
    d = pathlib.Path(args.dir)
    meta = json.loads((d / "gemm.json").read_text())
    J, M, N = meta["jobs"], meta["M"], meta["N"]
    t = np.fromfile(d / "tiles" / "d.bin", np.uint32).reshape(J, M // TM, N // TN, TM, TN)
    t.transpose(0, 1, 3, 2, 4).reshape(J, M, N).tofile(d / "tiles" / "gemm_d.bin")


def pack4(x: np.ndarray) -> np.ndarray:
    """E2M1 elements, one per byte, two per byte: element 2i in bits 3:0, 2i + 1 in 7:4."""
    return ((x[..., 0::2] & 0xF) | ((x[..., 1::2] & 0xF) << 4)).astype(np.uint8)


def scale_beats(sf: np.ndarray, group: int, S: int, nb: int) -> np.ndarray:
    """One 32-byte beat per (group of rows, step): scale j of row g * group + r in byte r * nb + j."""
    rows = sf.shape[0]
    beats = np.zeros((rows // group, S, 32), np.uint8)
    v = sf.reshape(rows // group, group, S, nb).transpose(0, 2, 1, 3).reshape(rows // group, S, group * nb)
    beats[:, :, : group * nb] = v
    return beats


def rtl(args) -> None:
    d = pathlib.Path(args.dir)
    meta, x = load(d)
    S, nb = meta["K"] // meta["k"], meta["blocks"]
    assert args.R * nb <= 32 and args.C * nb <= 32
    with open(d / "in.bin", "wb") as f:
        for j in range(meta["jobs"]):
            f.write(pack4(x["a"][j]).tobytes() + pack4(x["b"][j]).tobytes() + x["c"][j].tobytes())
            f.write(scale_beats(x["sfa"][j], args.R, S, nb).tobytes())
            f.write(scale_beats(x["sfb"][j], args.C, S, nb).tobytes())
    (d / "expected.bin").write_bytes((d / "tiles" / "gemm_d.bin").read_bytes())
    print(json.dumps({"jobs": meta["jobs"], "in_bytes": (d / "in.bin").stat().st_size}))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--inst", required=True)
    g.add_argument("--jobs", type=int, default=96)
    g.add_argument("--M", type=int, default=64)
    g.add_argument("--N", type=int, default=64)
    g.add_argument("--K", type=int, default=256)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True)
    for name in ("tiles", "gather"):
        sub.add_parser(name).add_argument("--dir", required=True)
    r = sub.add_parser("rtl")
    r.add_argument("--dir", required=True)
    r.add_argument("--R", type=int, default=4)
    r.add_argument("--C", type=int, default=4)
    args = ap.parse_args()
    {"gen": gen, "tiles": tiles, "gather": gather, "rtl": rtl}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
