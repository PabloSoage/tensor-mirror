"""Cases for the block-scale harness (gpu/mma_block_scale.cu), its scale-layout probe, and the check.

    python tools/bs_cases.py gen   --inst m16n8k32.mxf8f6f4.1x.f32.e4m3.e2m1.f32.ue8m0 --tiles 2048 --out DIR
    mma_block_scale <inst> <tiles> DIR/a.bin DIR/b.bin DIR/c.bin DIR/sa.bin DIR/sb.bin DIR/sel.bin DIR/d.bin
    python tools/bs_cases.py check --dir DIR [--mmasim] [--jobs N]
    python tools/bs_cases.py probe-gen --inst ... --out DIR      (then run as above)
    python tools/bs_cases.py probe-check --dir DIR

The scale factors are kept as matrices, SF_A (M x blocks) and SF_B (blocks x N), in
``sfa.bin`` and ``sfb.bin``. ``scale_regs`` places them in each lane's scale-a-data and
scale-b-data words as H-010 predicts from the PTX ISA's text; bytes and lanes that hold no
scale factor are random. The probe checks that placement: each lane and byte of the
words gets a different value, a single element per row (or column) is non-zero, and D
then says which lane and byte the GPU took for every SF_A[r][j] and SF_B[j][n].

``gen`` tiles, by thirds: everything raw (elements, C, scales, the unused bits; overflow,
underflow and NaN scales are common), raw elements with scales near 1, and all values near
1. Every tile has random valid selectors. The directed cases of ``directed`` follow.
"""

import argparse
import json
import multiprocessing
import pathlib
import random
import struct
import sys

import numpy as np

from tensormirror.blockscale import block_scale, mma

ONE = {"e4m3": 0x38, "e5m2": 0x3C, "e3m2": 0x0C, "e2m3": 0x08, "e2m1": 0x08, "packed": 0x2}
SCALE_ONE = {"ue8m0": 127, "ue4m3": 0x38}
SIGN = {"e4m3": 0x80, "e5m2": 0x80, "e3m2": 0x20, "e2m3": 0x20, "e2m1": 0x20}


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def parse(inst: str) -> dict:
    shape, kind, vec, td, ta, tb, tc, stype = inst.split(".")
    K, blocks = int(shape.split("k")[1]), int(vec[0])
    return {"shape": shape, "kind": kind, "vec": vec, "ta": ta, "tb": tb, "stype": stype, "K": K,
            "blocks": blocks, "block": K // blocks, "packed": K == 64}


def selectors(rng: random.Random, blocks: int) -> list[int]:
    """byte-id-a, thread-id-a, byte-id-b, thread-id-b, valid for the scale_vec (PTX ISA Table 46)."""
    byte = {1: [0, 1, 2, 3], 2: [0, 2], 4: [0]}[blocks]
    return [rng.choice(byte), rng.randint(0, 1), rng.choice(byte), rng.randint(0, 3)]


def a_lane(r: int, tid_a: int) -> int:
    return 4 * (r % 8) + 2 * tid_a + r // 8  # H-010, prediction 1


def b_lane(n: int, tid_b: int) -> int:
    return 4 * n + tid_b


def scale_regs(sfa, sfb, sel, rng: np.random.Generator):
    """Each lane's scale-a-data and scale-b-data words for one tile (H-010, prediction 1)."""
    bida, tida, bidb, tidb = sel
    blocks = sfa.shape[1]
    sa = rng.integers(0, 256, (32, 4), dtype=np.uint8)
    sb = rng.integers(0, 256, (32, 4), dtype=np.uint8)
    for r in range(16):
        for j in range(blocks):
            sa[a_lane(r, tida), bida + j] = sfa[r, j]
    for n in range(8):
        for j in range(blocks):
            sb[b_lane(n, tidb), bidb + j] = sfb[j, n]
    return sa.view("<u4").reshape(32), sb.view("<u4").reshape(32)


def element_bits(ins: dict, t: str) -> tuple[int, int, int]:
    """(lsb, width, exponent bits) of an element inside its byte as the harness takes it."""
    if ins["packed"]:
        return 0, 4, 2
    return {"e4m3": (0, 8, 4), "e5m2": (0, 8, 5), "e3m2": (0, 6, 3), "e2m3": (0, 6, 2), "e2m1": (2, 4, 2)}[t]


def near_one(rng: random.Random, ins: dict, t: str) -> int:
    """A byte holding an element within a binade or so of 1, random in the unused bits."""
    lsb, width, eb = element_bits(ins, t)
    mb = width - 1 - eb
    bias = (1 << (eb - 1)) - 1
    e = min(max(bias + rng.randint(-1, 1), 1), (1 << eb) - (2 if t == "e5m2" else 1))
    frac = rng.getrandbits(mb)
    if t == "e4m3" and e == 15:
        frac &= 0b110  # not the NaN pattern
    field = (rng.getrandbits(1) << (width - 1)) | (e << mb) | frac
    return (field << lsb) | (rng.getrandbits(8) & ~(((1 << width) - 1) << lsb) & 0xFF)


def near_one_scale(rng: random.Random, stype: str) -> int:
    if stype == "ue8m0":
        return 127 + rng.randint(-8, 8)
    return (rng.getrandbits(1) << 7) | rng.randint(0x30, 0x48)  # sign bit random (prediction 4)


def near_one_c(rng: random.Random) -> int:
    return f32(rng.choice([-1, 1]) * (1 + rng.random()) * 2.0 ** rng.randint(-3, 3))


def directed(ins: dict) -> list:
    """(name, a row 0, b column 0, SF_A row 0, SF_B column 0, c[0][0]); elements 1.0 unless given."""
    one = ONE["packed" if ins["packed"] else ins["ta"]]
    oneb = ONE["packed" if ins["packed"] else ins["tb"]]
    K, nb, st = ins["K"], ins["blocks"], ins["stype"]
    ones_a, ones_b = [one] * K, [oneb] * K
    cases = []
    if st == "ue8m0":
        # K products of 2**-150 (1.0 x 1.0 x 2**-75 x 2**-75) sum to K * 2**-150, which f32
        # holds; a zero C at e_zero must not truncate them
        cases.append(("tiny_products_c_zero", ones_a, ones_b, [52] * nb, [52] * nb, 0))
        cases.append(("overflow_big_scales", ones_a, ones_b, [254] * nb, [254] * nb, 0))
        cases.append(("overflow_just", [one] + [0] * (K - 1), [oneb] + [0] * (K - 1), [191] * nb, [192] * nb, 0))
        cases.append(("nan_scale_a", ones_a, ones_b, [0xFF] * nb, [127] * nb, 0))
    else:
        cases.append(("ue4m3_sign_bit", ones_a, ones_b, [0xB8] * nb, [0x38] * nb, 0))
        cases.append(("ue4m3_neg_zero_scale", ones_a, ones_b, [0x80] * nb, [0x38] * nb, f32(1.0)))
        cases.append(("ue4m3_nan", ones_a, ones_b, [0x7F] * nb, [0x38] * nb, 0))
        cases.append(("ue4m3_nan_with_sign", ones_a, ones_b, [0xFF] * nb, [0x38] * nb, 0))
        cases.append(("ue4m3_zero_scale_c_neg_zero", ones_a, ones_b, [0x00] * nb, [0x38] * nb, f32(-0.0)))
        cases.append(("ue4m3_subnormal_scales", ones_a, ones_b, [0x01] * nb, [0x03] * nb, 0))
    neg = one | (0x8 if ins["packed"] else SIGN[ins["ta"]])
    cases.append(("cancel_to_zero_c_min_subnormal", [one] * (K // 2) + [neg] * (K // 2),
                  ones_b, [SCALE_ONE[st]] * nb, [SCALE_ONE[st]] * nb, 1))
    return cases


def gen(args) -> None:
    ins = parse(args.inst)
    K, nb = ins["K"], ins["blocks"]
    cases = directed(ins)
    n, nd = args.tiles, len(cases)
    nt = n + nd
    nprng = np.random.default_rng(args.seed)
    rng = random.Random(args.seed)
    a = nprng.integers(0, 256, (nt, 16, K), dtype=np.uint8)
    b = nprng.integers(0, 256, (nt, 8, K), dtype=np.uint8)
    c = nprng.integers(0, 1 << 32, (nt, 16, 8), dtype=np.uint64).astype(np.uint32)
    sfa = nprng.integers(0, 256, (nt, 16, nb), dtype=np.uint8)
    sfb = nprng.integers(0, 256, (nt, nb, 8), dtype=np.uint8)
    third = n // 3
    for t in range(third, n):  # scales near 1
        sfa[t] = [[near_one_scale(rng, ins["stype"]) for _ in range(nb)] for _ in range(16)]
        sfb[t] = [[near_one_scale(rng, ins["stype"]) for _ in range(8)] for _ in range(nb)]
    for t in range(2 * third, n):  # and elements and C near 1
        a[t] = [[near_one(rng, ins, ins["ta"]) for _ in range(K)] for _ in range(16)]
        b[t] = [[near_one(rng, ins, ins["tb"]) for _ in range(K)] for _ in range(8)]
        c[t] = [[near_one_c(rng) for _ in range(8)] for _ in range(16)]
    for i, (_, av, bv, sav, sbv, cv) in enumerate(cases):
        t = n + i
        a[t], b[t], c[t] = 0, 0, 0
        sfa[t], sfb[t] = SCALE_ONE[ins["stype"]], SCALE_ONE[ins["stype"]]
        a[t, 0], b[t, 0], sfa[t, 0], sfb[t, :, 0], c[t, 0, 0] = av, bv, sav, sbv, cv
    sel = np.array([selectors(rng, nb) for _ in range(nt)], dtype=np.uint16)
    sa = np.empty((nt, 32), np.uint32)
    sb = np.empty((nt, 32), np.uint32)
    for t in range(nt):
        sa[t], sb[t] = scale_regs(sfa[t], sfb[t], sel[t], nprng)
    write(pathlib.Path(args.out), a=a, b=b, c=c, sfa=sfa, sfb=sfb, sa=sa, sb=sb, sel=sel,
          meta={"inst": args.inst, "tiles": nt, "random_tiles": n, "thirds": third,
                "directed": [x[0] for x in cases], "seed": args.seed})


def write(out: pathlib.Path, meta: dict, **arrays) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, x in arrays.items():
        x.tofile(out / f"{name}.bin")
    (out / "cases.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "probe"}))


def load(d_dir: pathlib.Path):
    meta = json.loads((d_dir / "cases.json").read_text())
    ins = parse(meta["inst"])
    nt, K, nb = meta["tiles"], ins["K"], ins["blocks"]
    x = {"a": np.fromfile(d_dir / "a.bin", np.uint8).reshape(nt, 16, K),
         "b": np.fromfile(d_dir / "b.bin", np.uint8).reshape(nt, 8, K),
         "c": np.fromfile(d_dir / "c.bin", np.uint32).reshape(nt, 16, 8),
         "sfa": np.fromfile(d_dir / "sfa.bin", np.uint8).reshape(nt, 16, nb),
         "sfb": np.fromfile(d_dir / "sfb.bin", np.uint8).reshape(nt, nb, 8)}
    return meta, ins, x


def _model_tiles(job):
    inst, a, b, c, sfa, sfb, measured = job
    p = block_scale(inst, measured)
    out = np.empty(c.shape, dtype=np.uint32)
    for t in range(c.shape[0]):
        out[t] = mma(p, a[t].tolist(), b[t].T.tolist(), c[t].tolist(), sfa[t].tolist(), sfb[t].tolist())
    return out


def model_output(inst: str, x: dict, jobs: int, measured: bool = True) -> np.ndarray:
    chunks = np.array_split(np.arange(x["c"].shape[0]), max(1, jobs * 4))
    work = [(inst, x["a"][ix], x["b"][ix], x["c"][ix], x["sfa"][ix], x["sfb"][ix], measured)
            for ix in chunks if len(ix)]
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            return np.concatenate(pool.map(_model_tiles, work))
    return np.concatenate([_model_tiles(w) for w in work])


def mmasim_name(ins: dict) -> str:
    return f"{ins['shape']}.block{ins['block']}.f32.{ins['ta']}.{ins['tb']}.f32.{ins['stype']}"


def mmasim_output(ins: dict, x: dict):
    """MMA-Sim's D, or None if it does not model the instruction."""
    import torch
    from mmasim.nv_ptx import MMABlockScale
    from mmasim.nv_ptx.isa_mma import arch_mma_block_scale_shape_and_type

    name = mmasim_name(ins)
    if name not in arch_mma_block_scale_shape_and_type["RTX Blackwell"]:
        return None
    sim = MMABlockScale("RTX Blackwell", name)
    tt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    st = {"ue8m0": torch.float8_e8m0fnu, "ue4m3": torch.float8_e4m3fn}[ins["stype"]]
    a, b = x["a"], x["b"]
    if ins["packed"]:  # MMA-Sim's unpacking takes element 2i from the high nibble
        a = ((a[..., 0::2] & 0xF) << 4 | (a[..., 1::2] & 0xF)).astype(np.uint8)
        b = ((b[..., 0::2] & 0xF) << 4 | (b[..., 1::2] & 0xF)).astype(np.uint8)
    M, N = 16, 8

    def as_torch(v, t):
        v = torch.from_numpy(np.ascontiguousarray(v))
        return v.view(tt[t]) if t in tt and not ins["packed"] else v

    A = as_torch(np.repeat(a[:, :, None, :], N, axis=2), ins["ta"])
    B = as_torch(np.repeat(b[:, None, :, :], M, axis=1), ins["tb"])
    C = torch.from_numpy(x["c"].view(np.int32)).view(torch.float32)
    alpha = torch.from_numpy(np.ascontiguousarray(np.repeat(x["sfa"][:, :, None, :], N, axis=2))).view(st)
    beta = torch.from_numpy(np.ascontiguousarray(np.repeat(x["sfb"].transpose(0, 2, 1)[:, None], M, axis=1))).view(st)
    s = sim.dpa(A, B, C, alpha, beta)
    return s.view(torch.int32).numpy().view(np.uint32)


def check(args) -> int:
    d_dir = pathlib.Path(args.dir)
    meta, ins, x = load(d_dir)
    d = np.fromfile(d_dir / "d.bin", np.uint32).reshape(x["c"].shape)
    model = model_output(meta["inst"], x, args.jobs)
    t3, n = meta["thirds"], meta["random_tiles"]
    report = {"inst": meta["inst"], "tiles": meta["tiles"], "elements": int(d.size),
              "gpu_vs_model": int((d != model).sum()),
              "raw_third": int((d[:t3] != model[:t3]).sum()),
              "raw_elements_third": int((d[t3:2 * t3] != model[t3:2 * t3]).sum()),
              "near_one_third": int((d[2 * t3:n] != model[2 * t3:n]).sum()),
              "gpu_inf": int(((d & 0x7FFFFFFF) == 0x7F800000).sum()),
              "gpu_max_finite": int(((d & 0x7FFFFFFF) == 0x7F7FFFFF).sum()),
              "gpu_nan": int(((d & 0x7FFFFFFF) > 0x7F800000).sum())}
    if args.mmasim:
        s = mmasim_output(ins, x)
        report["mmasim_name"] = mmasim_name(ins)
        report["gpu_vs_mmasim"] = None if s is None else int((d != s).sum())
        report["model_vs_mmasim"] = None if s is None else int((model != s).sum())
    report["directed"] = {
        name: {"gpu": hex(d[n + k, 0, 0]), "model": hex(model[n + k, 0, 0]),
               "rest_of_tile_ok": bool((d[n + k] == model[n + k]).all())}
        for k, name in enumerate(meta["directed"])}
    bad = np.argwhere(d != model)[:10]
    report["first_mismatches"] = [
        {"tile": int(t), "i": int(i), "j": int(j), "a": x["a"][t, i].tolist(), "b": x["b"][t, j].tolist(),
         "sfa": x["sfa"][t, i].tolist(), "sfb": x["sfb"][t, :, j].tolist(), "c": hex(x["c"][t, i, j]),
         "gpu": hex(d[t, i, j]), "model": hex(model[t, i, j])} for t, i, j in bad]
    (d_dir / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("first_mismatches",)}))
    return 1 if report["gpu_vs_model"] else 0


# ---- the scale-layout probe -------------------------------------------------------------------
def probe_code(stype: str, lane: int, byte: int, half: int) -> int:
    """The scale value written at (lane, byte): distinct for every lane and byte (of a half)."""
    if stype == "ue8m0":
        return 63 + 4 * lane + byte  # 2**-64 .. 2**63
    return 1 + 4 * (lane % 16) + byte if lane // 16 == half else 0  # ue4m3: 64 codes, 0 elsewhere


def probe_value(stype: str, code: int) -> float:
    if stype == "ue8m0":
        return 2.0 ** (code - 127)
    e, m = code >> 3, code & 7
    return (m / 8) * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


def probe_gen(args) -> None:
    ins = parse(args.inst)
    K, nb, bs, st = ins["K"], ins["blocks"], ins["block"], ins["stype"]
    one_a = ONE["packed" if ins["packed"] else ins["ta"]]
    one_b = ONE["packed" if ins["packed"] else ins["tb"]]
    byte = {1: [0, 1, 2, 3], 2: [0, 2], 4: [0]}[nb]
    jobs = [(side, half, j, (ba, ta, bb, tb)) for side in "ab" for half in ((0, 1) if st == "ue4m3" else (0,))
            for j in range(nb) for ba in byte for ta in (0, 1) for bb in byte for tb in range(4)]
    nt = len(jobs)
    a = np.zeros((nt, 16, K), np.uint8)
    b = np.zeros((nt, 8, K), np.uint8)
    c = np.zeros((nt, 16, 8), np.uint32)
    sa = np.zeros((nt, 32, 4), np.uint8)
    sb = np.zeros((nt, 32, 4), np.uint8)
    for t, (side, half, j, _) in enumerate(jobs):
        codes = np.array([[probe_code(st, lane, y, half) for y in range(4)] for lane in range(32)], np.uint8)
        if side == "a":  # one element per row of A, in block j; B all ones; SF_B all one
            a[t, :, j * bs], b[t] = one_a, one_b
            sa[t], sb[t] = codes, SCALE_ONE[st]
        else:
            a[t], b[t, :, j * bs] = one_a, one_b
            sa[t], sb[t] = SCALE_ONE[st], codes
    sel = np.array([s for *_, s in jobs], dtype=np.uint16)
    write(pathlib.Path(args.out), a=a, b=b, c=c, sa=sa.view("<u4").reshape(nt, 32), sb=sb.view("<u4").reshape(nt, 32),
          sel=sel, sfa=np.zeros((nt, 16, nb), np.uint8), sfb=np.zeros((nt, nb, 8), np.uint8),
          meta={"inst": args.inst, "tiles": nt, "random_tiles": nt, "thirds": nt, "directed": [], "seed": 0,
                "probe": [list(x[:3]) + [list(x[3])] for x in jobs]})


def probe_check(args) -> int:
    d_dir = pathlib.Path(args.dir)
    meta = json.loads((d_dir / "cases.json").read_text())
    ins = parse(meta["inst"])
    st = ins["stype"]
    d = np.fromfile(d_dir / "d.bin", np.float32).reshape(meta["tiles"], 16, 8)
    lookup = {}
    for half in (0, 1):
        for lane in range(32):
            for y in range(4):
                code = probe_code(st, lane, y, half)
                if code:
                    lookup[(half, probe_value(st, code))] = (lane, y)
    stats = {"a": [0, 0], "b": [0, 0]}  # as predicted, not
    other = []
    for t, (side, half, j, (ba, ta, bb, tb)) in enumerate(meta["probe"]):
        for r in range(16):
            for n in range(8):
                v = float(d[t, r, n])
                want = (a_lane(r, ta), ba + j) if side == "a" else (b_lane(n, tb), bb + j)
                if st == "ue4m3" and want[0] // 16 != half:
                    ok = v == 0.0
                    got = "zero" if v == 0.0 else lookup.get((half, v), v)
                else:
                    got = lookup.get((half, v), v)
                    ok = got == want
                stats[side][0 if ok else 1] += 1
                if not ok and len(other) < 20:
                    other.append({"side": side, "block": j, "sel": [ba, ta, bb, tb], "r": r, "n": n,
                                  "predicted_lane_byte": want, "got": got})
    report = {"inst": meta["inst"], "tiles": meta["tiles"],
              "sf_a": {"as_predicted": stats["a"][0], "not": stats["a"][1]},
              "sf_b": {"as_predicted": stats["b"][0], "not": stats["b"][1]}, "first_other": other}
    (d_dir / "probe.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, default=str))
    return 1 if stats["a"][1] or stats["b"][1] else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "probe-gen"):
        g = sub.add_parser(name)
        g.add_argument("--inst", required=True)
        g.add_argument("--tiles", type=int, default=2048)
        g.add_argument("--seed", type=int, default=1)
        g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--mmasim", action="store_true")
    c.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("probe-check")
    p.add_argument("--dir", required=True)
    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args)
        return 0
    if args.cmd == "probe-gen":
        probe_gen(args)
        return 0
    if args.cmd == "probe-check":
        return probe_check(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
