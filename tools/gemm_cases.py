"""GEMM jobs for rtl/tm_pl_array.v and their reference: a loop of mma.sync per 16 x 8 tile.

    python tools/gemm_cases.py gen    --arch Ampere --types f16.f16.f16.f16 --jobs 96 --out DIR
    python tools/gemm_cases.py tiles  --dir DIR --inst m16n8k16.f16.f16.f16.f16
    mma_sync <inst> <tiles> DIR/<inst>/a.bin ... d.bin <steps>      (on the GPU host)
    python tools/gemm_cases.py gather --dir DIR --inst m16n8k16.f16.f16.f16.f16
    python tools/gemm_cases.py rtl    --dir DIR --ref <inst>          (in.bin, expected.bin)
    tb_tm_pl_array DIR/in.bin DIR/expected.bin DIR/rtl/d.bin ...
    python tools/gemm_cases.py compare --dir DIR --x <inst> --y rtl [--model]

A job is D = C + A B, with A M x K, B^T N x K and C M x N, raw words (``gen``). The thirds
of the jobs are uniform raw bits, operands within 2 binades of 1, and operands within 8
binades of 1 (so f16 reaches its subnormals and infinities); bits a format does not use are
random. ``tiles`` cuts every job into 16 x 8 output tiles over the whole of K, in the files
of tools/gpu_cases.py (so its ``check`` runs on them too), for gpu/mma_sync.cu with
steps = K / k: each warp chains K in order, as a GEMM kernel does. ``gather`` puts the
tiles' d.bin back as DIR/<inst>/D.bin, jobs x M x N. ``rtl`` writes the beats of
rtl/tm_pl_array.v (A, B^T, C of each job, packed) and the expected D beats from one
reference. ``compare`` counts the elements where two D differ, per third, and the finite
ones; ``--model`` adds the integer model chained instruction after instruction.
"""

import argparse
import json
import multiprocessing
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gpu_cases import WORD, parse, scaled_word  # noqa: E402

from tensormirror.arch import PTX_TYPES, tfdpa  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402

TM, TN = 16, 8  # the output tile of one warp (m16n8kK)
SPREADS = (None, 2, 8)  # the thirds: raw bits, then two spreads in binades


def types_of(meta: dict) -> tuple[str, str, str, str]:
    td, ta, tb, tc = meta["types"].split(".")
    return td, ta, tb, tc


def size(t: str) -> int:
    return PTX_TYPES[t][1]


def gen(args) -> None:
    td, ta, tb, tc = args.types.split(".")
    J, M, N, K = args.jobs, args.M, args.N, args.K
    assert M % TM == 0 and N % TN == 0 and J % 3 == 0
    nprng = np.random.default_rng(args.seed)
    rng = random.Random(args.seed)

    def raw(shape, t):
        return nprng.integers(0, 1 << (8 * size(t)), size=shape, dtype=np.uint64).astype(WORD[size(t)])

    a, b, c = raw((J, M, K), ta), raw((J, N, K), tb), raw((J, M, N), tc)
    third = J // 3
    for j in range(third, J):
        sp = SPREADS[j // third]
        fa, fb, fc = PTX_TYPES[ta][0], PTX_TYPES[tb][0], PTX_TYPES[tc][0]
        a[j] = [[scaled_word(rng, fa, size(ta), sp) for _ in range(K)] for _ in range(M)]
        b[j] = [[scaled_word(rng, fb, size(tb), sp) for _ in range(K)] for _ in range(N)]
        c[j] = [[scaled_word(rng, fc, size(tc), sp + 1) for _ in range(N)] for _ in range(M)]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("A", a), ("B", b), ("C", c)):
        x.tofile(out / f"{name}.bin")
    meta = {"arch": args.arch, "types": args.types, "jobs": J, "M": M, "N": N, "K": K, "seed": args.seed,
            "thirds": [list(SPREADS), third]}
    (out / "gemm.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def load(d: pathlib.Path):
    meta = json.loads((d / "gemm.json").read_text())
    td, ta, tb, tc = types_of(meta)
    J, M, N, K = meta["jobs"], meta["M"], meta["N"], meta["K"]
    a = np.fromfile(d / "A.bin", WORD[size(ta)]).reshape(J, M, K)
    b = np.fromfile(d / "B.bin", WORD[size(tb)]).reshape(J, N, K)
    c = np.fromfile(d / "C.bin", WORD[size(tc)]).reshape(J, M, N)
    return meta, a, b, c


def tiles(args) -> None:
    d = pathlib.Path(args.dir)
    meta, a, b, c = load(d)
    ins = parse(args.inst)
    assert ins["types"] == meta["types"] and (ins["M"], ins["N"]) == (TM, TN)
    J, M, N, K = meta["jobs"], meta["M"], meta["N"], meta["K"]
    steps = K // ins["K"]
    # tile (job, tm, tn): rows tm*16.., columns tn*8.., the whole of K
    ta = a.reshape(J, M // TM, 1, TM, K).repeat(N // TN, axis=2).reshape(-1, TM, K)
    tb = b.reshape(J, 1, N // TN, TN, K).repeat(M // TM, axis=1).reshape(-1, TN, K)
    tc = c.reshape(J, M // TM, TM, N // TN, TN).transpose(0, 1, 3, 2, 4).reshape(-1, TM, TN)
    out = d / args.inst
    out.mkdir(exist_ok=True)
    for name, x in (("a", ta), ("b", tb), ("c", tc)):
        np.ascontiguousarray(x).tofile(out / f"{name}.bin")
    nt = len(tc)
    tmeta = {"arch": meta["arch"], "inst": args.inst, "random_tiles": nt, "scaled_from": nt // 3,
             "directed": [], "tiles": nt, "seed": meta["seed"], "steps": steps}
    (out / "cases.json").write_text(json.dumps(tmeta, indent=1) + "\n")
    print(json.dumps({"tiles": nt, "steps": steps}))


def gather(args) -> None:
    d = pathlib.Path(args.dir)
    meta = json.loads((d / "gemm.json").read_text())
    td = types_of(meta)[0]
    J, M, N = meta["jobs"], meta["M"], meta["N"]
    t = np.fromfile(d / args.inst / "d.bin", WORD[size(td)]).reshape(J, M // TM, N // TN, TM, TN)
    t.transpose(0, 1, 3, 2, 4).reshape(J, M, N).tofile(d / args.inst / "D.bin")


def rtl(args) -> None:
    d = pathlib.Path(args.dir)
    meta, a, b, c = load(d)
    with open(d / "in.bin", "wb") as f:
        for j in range(meta["jobs"]):
            f.write(a[j].tobytes() + b[j].tobytes() + c[j].tobytes())
    (d / "expected.bin").write_bytes((d / args.ref / "D.bin").read_bytes())
    print(json.dumps({"jobs": meta["jobs"], "in_bytes": (d / "in.bin").stat().st_size, "ref": args.ref}))


def _model_jobs(job):
    arch, types, k, a, b, c = job
    p = tfdpa(arch, types)
    J, M, N = c.shape
    out = np.empty((J, M, N), dtype=np.uint64)
    for t in range(J):
        for i in range(M):
            ai = a[t, i].tolist()
            for j in range(N):
                bj, acc = b[t, j].tolist(), int(c[t, i, j])
                for k0 in range(0, len(ai), k):  # one instruction after another
                    acc = dot(p, ai[k0 : k0 + k], bj[k0 : k0 + k], acc)
                out[t, i, j] = acc
    return out


def model(meta: dict, a, b, c, k: int, jobs: int) -> np.ndarray:
    td = types_of(meta)[0]
    rows = [(j, i) for j in range(meta["jobs"]) for i in range(0, meta["M"], 4)]
    work = [(meta["arch"], meta["types"], k, a[j : j + 1, i : i + 4], b[j : j + 1], c[j : j + 1, i : i + 4])
            for j, i in rows]
    with multiprocessing.Pool(jobs) as pool:
        parts = pool.map(_model_jobs, work)
    out = np.empty(c.shape, dtype=np.uint64)
    for (j, i), p in zip(rows, parts, strict=True):
        out[j, i : i + 4] = p[0]
    return out.astype(WORD[size(td)])


def model_cmd(args) -> None:
    """The model's D as a reference directory, DIR/model/D.bin, for k-wide instructions."""
    d = pathlib.Path(args.dir)
    meta, a, b, c = load(d)
    (d / "model").mkdir(exist_ok=True)
    model(meta, a, b, c, args.k, args.jobs).tofile(d / "model" / "D.bin")


def finite(x: np.ndarray, t: str) -> np.ndarray:
    f = PTX_TYPES[t][0]
    e = (x.astype(np.uint64) >> f.mbits) & ((1 << f.ebits) - 1)
    return e != (1 << f.ebits) - 1


def compare(args) -> int:
    d = pathlib.Path(args.dir)
    meta, a, b, c = load(d)
    td = types_of(meta)[0]
    J, M, N = meta["jobs"], meta["M"], meta["N"]

    def read(name):
        return np.fromfile(d / name / "D.bin", WORD[size(td)]).reshape(J, M, N)

    third = meta["thirds"][1]
    ds = {args.x: read(args.x), args.y: read(args.y)}
    if args.model:
        k = parse(args.x)["K"] if args.x != "rtl" else parse(args.y)["K"]
        ds["model"] = model(meta, a, b, c, k, args.jobs)
        ds["model"].tofile(d / "model-D.bin")
    names = list(ds)
    x = ds[args.x]
    report = {"arch": meta["arch"], "types": meta["types"], "jobs": J, "elements": int(x.size),
              "finite": [int(finite(x[s * third : (s + 1) * third], td).sum()) for s in range(3)], "pairs": {}}
    for i, p in enumerate(names):
        for q in names[i + 1 :]:
            diff = ds[p] != ds[q]
            report["pairs"][f"{p} vs {q}"] = {
                "mismatches": int(diff.sum()),
                "per_third": [int(diff[s * third : (s + 1) * third].sum()) for s in range(3)]}
    (d / f"compare-{args.x}-{args.y}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 1 if any(v["mismatches"] for v in report["pairs"].values()) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--arch", default="Ampere")
    g.add_argument("--types", default="f16.f16.f16.f16")
    g.add_argument("--jobs", type=int, default=96)
    g.add_argument("--M", type=int, default=64)
    g.add_argument("--N", type=int, default=64)
    g.add_argument("--K", type=int, default=256)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True)
    for name in ("tiles", "gather"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--inst", required=True)
    s = sub.add_parser("rtl")
    s.add_argument("--dir", required=True)
    s.add_argument("--ref", required=True, help="the directory under DIR whose D.bin is expected")
    s = sub.add_parser("model")
    s.add_argument("--dir", required=True)
    s.add_argument("--k", type=int, required=True, help="k of the instruction the loop runs")
    s.add_argument("--jobs", type=int, default=16)
    s = sub.add_parser("compare")
    s.add_argument("--dir", required=True)
    s.add_argument("--x", required=True)
    s.add_argument("--y", required=True)
    s.add_argument("--model", action="store_true")
    s.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args)
    elif args.cmd == "tiles":
        tiles(args)
    elif args.cmd == "gather":
        gather(args)
    elif args.cmd == "rtl":
        rtl(args)
    elif args.cmd == "model":
        model_cmd(args)
    else:
        return compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
