"""Show mismatching dot products of a checked case directory next to the exact value and
the model's result for several F, to see what the GPU does when no T-FDPA fits.

    python tools/inspect_cases.py <case dir> [n] [F list, e.g. 22,23,24,25]
"""

import dataclasses
import json
import pathlib
import struct
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

from tensormirror.arch import PTX_TYPES, tfdpa  # noqa: E402
from tensormirror.formats import NUM, decode  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402


def exact(fmt, bits):
    x = decode(fmt, bits)
    if x.kind != NUM:
        return None
    v = Fraction(x.sig) * Fraction(2) ** (x.exp - fmt.mbits)
    return -v if x.neg else v


def as_float(fmt, bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0] if fmt.width == 32 else \
        struct.unpack("<e", struct.pack("<H", bits))[0]


d_dir = pathlib.Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
Fs = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "22,23,24,25").split(",")]
meta = json.loads((d_dir / "cases.json").read_text())
ins, a, b, c = gpu_cases.load(d_dir, meta)
d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
base = tfdpa(meta["arch"], ins["types"], ins["shape"])
fa, fc, fd = PTX_TYPES[ins["ta"]][0], PTX_TYPES[ins["tc"]][0], PTX_TYPES[ins["td"]][0]
model = gpu_cases.model_output(meta["arch"], ins, a, b, c, jobs=8)
shown = 0
for t, i, j in np.argwhere(d != model):
    if t < meta["scaled_from"]:
        continue  # the scaled tiles are the readable ones
    av, bv = a[t, i].tolist(), b[t, j].tolist()
    prods = [exact(fa, x) * exact(fa, y) for x, y in zip(av, bv, strict=True)]
    cv = exact(fc, int(c[t, i, j]))
    if None in prods or cv is None:
        continue
    total = sum(prods) + cv
    print(f"tile {t} ({i},{j}) c={float(cv):.9g} products={[float(p) for p in prods]}")
    print(f"   exact={float(total):.12g} gpu={as_float(fd, int(d[t, i, j])):.12g} ({hex(d[t, i, j])})")
    for F in Fs:
        p = dataclasses.replace(base, F=F)
        r = dot(p, av, bv, int(c[t, i, j]))
        print(f"   F={F}: {as_float(fd, r):.12g} ({hex(r)}){'  <- gpu' if r == int(d[t, i, j]) else ''}")
    shown += 1
    if shown >= n:
        break
