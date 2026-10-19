"""One-hot probe of how kind::f8f6f4 reads an E2M1 A operand on RTX Blackwell.

Instruction m16n8k32.f8f6f4.f32.e2m1.e4m3.f32: B is E4M3, whose layout and arithmetic
already match the model, and B[k][n] = value(k) is different for every k (and the same
for every n). Each tile has a single non-zero byte in A, at (row, k), holding one of the
candidate encodings of +1.0. D[row][n] then reads back which k the GPU paired it with
and what value it gave it.

    python tools/probe_fp4.py gen <dir>            (then run it with gpu_batch.sh run)
    python tools/probe_fp4.py analyze <dir>
"""

import itertools
import json
import pathlib
import struct
import sys

import numpy as np

INST = "m16n8k32.f8f6f4.f32.e2m1.e4m3.f32"
# +1.0 in E2M1 is 0b0010; candidate placements in the byte, and garbage in the rest
CANDIDATES = {"bits5:2": 0x08, "bits3:0": 0x02, "bits7:4": 0x20, "bits5:2+pad": 0xC9, "bits3:0+pad": 0xF2}


def e4m3(e: int, m: int) -> int:
    return ((e + 7) << 3) | m


def b_value(k: int) -> float:
    e, m = k // 4 - 3, (k % 4) * 2  # 32 distinct values: exponents -3..4, mantissas 0,2,4,6
    return (1 + m / 8) * 2.0**e


def b_byte(k: int) -> int:
    return e4m3(k // 4 - 3, (k % 4) * 2)


def gen(out: pathlib.Path) -> None:
    jobs = list(itertools.product(CANDIDATES, range(16), range(32)))
    nt = len(jobs)
    a = np.zeros((nt, 16, 32), np.uint8)
    b = np.zeros((nt, 8, 32), np.uint8)
    b[:] = [b_byte(k) for k in range(32)]
    c = np.zeros((nt, 16, 8), np.uint32)
    for t, (cand, r, k) in enumerate(jobs):
        a[t, r, k] = CANDIDATES[cand]
    out.mkdir(parents=True, exist_ok=True)
    a.tofile(out / "a.bin")
    b.tofile(out / "b.bin")
    c.tofile(out / "c.bin")
    meta = {"arch": "RTX Blackwell", "inst": INST, "tiles": nt, "random_tiles": nt, "scaled_from": nt,
            "directed": [], "seed": 0, "probe": [list(j) for j in jobs]}
    (out / "cases.json").write_text(json.dumps(meta) + "\n")


def analyze(out: pathlib.Path) -> None:
    meta = json.loads((out / "cases.json").read_text())
    d = np.fromfile(out / "d.bin", np.uint32).reshape(meta["tiles"], 16, 8)
    summary = {}
    for t, (cand, r, k) in enumerate(meta["probe"]):
        f = [struct.unpack("<f", struct.pack("<I", int(x)))[0] for x in d[t].ravel()]
        f = np.array(f).reshape(16, 8)
        s = summary.setdefault(cand, {"tiles": 0, "exactly_value_k_in_row": 0, "all_zero": 0, "other": []})
        s["tiles"] += 1
        expected = np.zeros((16, 8))
        expected[r] = b_value(k)
        if (f == expected).all():
            s["exactly_value_k_in_row"] += 1
        elif (f == 0).all():
            s["all_zero"] += 1
        elif len(s["other"]) < 4:
            nz = np.argwhere(f != 0)
            s["other"].append({"row": r, "k": k, "nonzero_at": nz[:4].tolist(), "values": f[f != 0][:4].tolist(),
                               "value_k": b_value(k)})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"gen": gen, "analyze": analyze}[sys.argv[1]](pathlib.Path(sys.argv[2]))
