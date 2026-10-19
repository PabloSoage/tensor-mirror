"""Laptop side of a board run of tm_pl (tools/kv260/tm_run.py on the KV260).

    python tools/kv260/board_cases.py pack  <vectors.txt> <cases.bin>
    python tools/kv260/board_cases.py check <vectors.txt> <results.bin> [report.json]

vectors.txt comes from tools/pl_vectors.py (a, b, c and the three expected d). pack
writes tm_pl's s_axis beats (32 bytes per case); check compares every result beat (16
bytes: d_f32, d_f32c16, d_f16 and 6 zero bytes) with the expected words, unit by unit.
"""

import json
import pathlib
import struct
import sys

UNITS = ("f32.f16.f16.f32", "f32.f16.f16.f16", "f16.f16.f16.f16")


def read(path):
    rows = []
    for line in pathlib.Path(path).read_text().split("\n"):
        if line.strip():
            rows.append([int(x, 16) for x in line.split()])
    return rows


def pack(vec, out):
    with open(out, "wb") as f:
        for r in read(vec):
            a, b, c = r[0:4], r[4:8], r[8]
            f.write(struct.pack("<4H4HI20x", *a, *b, c))


def check(vec, res, report=None):
    rows = read(vec)
    data = pathlib.Path(res).read_bytes()
    n = len(data) // 16
    bad = {u: 0 for u in UNITS}
    first = []
    nonzero_pad = 0
    for i in range(min(n, len(rows))):
        d0, d1, d2, pad16, pad32 = struct.unpack_from("<IIHHI", data, 16 * i)
        nonzero_pad += bool(pad16 or pad32)
        for u, got, want in zip(UNITS, (d0, d1, d2), rows[i][9:12], strict=True):
            if got != want:
                bad[u] += 1
                if len(first) < 10:
                    first.append({"case": i, "unit": u, "a": [hex(x) for x in rows[i][0:4]],
                                  "b": [hex(x) for x in rows[i][4:8]], "c": hex(rows[i][8]),
                                  "board": hex(got), "model": hex(want)})
    out = {"cases": len(rows), "results": n, "mismatches": bad, "nonzero_padding": nonzero_pad,
           "first_mismatches": first}
    print(json.dumps({k: v for k, v in out.items() if k != "first_mismatches"}))
    if report:
        pathlib.Path(report).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if n == len(rows) and not any(bad.values()) and not nonzero_pad else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pack":
        pack(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(check(*sys.argv[2:5]))
