"""Laptop side of a board run of tm_pl_x (tools/kv260/tm_run.py on the KV260).

    python tools/kv260/board_cases_x.py gpu   <GPU run dir> <vectors.txt>
    python tools/kv260/board_cases_x.py plain <GPU run dir> <vectors.txt>
    python tools/kv260/board_cases_x.py pack  <vectors.txt> <L> <AW> <BW> <NS> <cases.bin>
    python tools/kv260/board_cases_x.py check <vectors.txt> <L> <NS> <results.bin> [report.json]

The vectors are tb/tb_tm_pl_x.cpp's lines: a[0..L-1] b[0..L-1] c sfa[0..NS-1] sfb[0..NS-1] d,
in hex, with NS scales per operand: 0 without block scaling, 1 for kind::mxf8f6f4 (ST), the
blocks of a kind::mxf4(nvf4) instruction (GST). ``gpu`` writes them from a block_scale run of
gpu/mma_block_scale.cu (tools/bs_cases.py: a.bin, b.bin, c.bin, sfa.bin, sfb.bin, d.bin), one
line per element of D, tile by tile, row by row: d is what the GPU gave. Packed E2M1 elements
are the low nibble of their byte, as the harness takes them. ``plain`` does the same for a
run of gpu/mma_sync.cu (tools/gpu_cases.py), whose K must be one group of the unit (L).
``pack`` writes tm_pl_x's s_axis beats, NB per case: the case's bits {sfb, sfa, c, b, a} from
the lowest, max(NS, 1) bytes per scale, padded with zeros to 32 * NB bytes. ``check``
compares every result beat (16 bytes: d and 12 zero bytes) with the vectors' d.
"""

import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gpu_cases  # noqa: E402
from bs_cases import load  # noqa: E402


def read(path, L, ns):
    rows = []
    for line in pathlib.Path(path).read_text().split("\n"):
        if line.strip():
            w = [int(x, 16) for x in line.split()]
            assert len(w) == 2 * L + 2 + 2 * ns, line
            rows.append(w)
    return rows


def gpu(run_dir, out):
    meta, ins, x = load(pathlib.Path(run_dir))
    nb, mask = ins["blocks"], 0xF if ins["packed"] else 0xFF
    d = np.fromfile(pathlib.Path(run_dir) / "d.bin", np.uint32).reshape(meta["tiles"], 16, 8)
    with open(out, "w", newline="\n") as f:
        for t in range(meta["tiles"]):
            for r in range(16):
                a = " ".join(f"{v & mask:x}" for v in x["a"][t, r])
                sa = " ".join(f"{x['sfa'][t, r, j]:x}" for j in range(nb))
                for n in range(8):
                    b = " ".join(f"{v & mask:x}" for v in x["b"][t, n])
                    sb = " ".join(f"{x['sfb'][t, j, n]:x}" for j in range(nb))
                    f.write(f"{a} {b} {x['c'][t, r, n]:x} {sa} {sb} {d[t, r, n]:x}\n")


def plain(run_dir, out):
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "cases.json").read_text())
    ins, a, b, c = gpu_cases.load(run_dir, meta)
    d = np.fromfile(run_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
    with open(out, "w", newline="\n") as f:
        for t in range(meta["tiles"]):
            for r in range(ins["M"]):
                ar = " ".join(f"{v:x}" for v in a[t, r])
                for n in range(ins["N"]):
                    br = " ".join(f"{v:x}" for v in b[t, n])
                    f.write(f"{ar} {br} {c[t, r, n]:x} {d[t, r, n]:x}\n")


def pack(vec, L, aw, bw, ns, out):
    fs = max(ns, 1)  # bytes per scale field
    nb = (L * aw + L * bw + 32 + 16 * fs + 255) // 256
    with open(out, "wb") as f:
        for w in read(vec, L, ns):
            v, pos = 0, 0
            for k in range(L):
                v |= w[k] << pos
                pos += aw
            for k in range(L):
                v |= w[L + k] << pos
                pos += bw
            v |= w[2 * L] << pos
            for j in range(ns):
                v |= w[2 * L + 1 + j] << pos + 32 + 8 * j
                v |= w[2 * L + 1 + ns + j] << pos + 32 + 8 * fs + 8 * j
            f.write(v.to_bytes(32 * nb, "little"))


def check(vec, L, ns, res, report=None):
    rows = read(vec, L, ns)
    data = pathlib.Path(res).read_bytes()
    n = len(data) // 16
    bad, nonzero_pad, first = 0, 0, []
    for i in range(min(n, len(rows))):
        got, p0, p1, p2 = struct.unpack_from("<IIII", data, 16 * i)
        nonzero_pad += bool(p0 or p1 or p2)
        if got != rows[i][-1]:
            bad += 1
            if len(first) < 10:
                first.append({"case": i, "line": " ".join(f"{v:x}" for v in rows[i][:-1]),
                              "board": hex(got), "expected": hex(rows[i][-1])})
    out = {"cases": len(rows), "results": n, "mismatches": bad, "nonzero_padding": nonzero_pad,
           "first_mismatches": first}
    print(json.dumps({k: v for k, v in out.items() if k != "first_mismatches"}))
    if report:
        pathlib.Path(report).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if n == len(rows) and not bad and not nonzero_pad else 1


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in ("gpu", "plain"):
        (gpu if cmd == "gpu" else plain)(*args)
        sys.exit(0)
    if cmd == "pack":
        pack(args[0], *map(int, args[1:5]), args[5])
        sys.exit(0)
    sys.exit(check(args[0], int(args[1]), int(args[2]), *args[3:5]))
