"""Where in the byte does kind::f8f6f4 read an FP4 (E2M1) operand?

Re-runs the integer model on measured RTX Blackwell cases with E2M1 placed at each bit
position of its byte (lsb 0 to 4), and with or without the unused bits, and counts the
mismatches with the GPU. Uses the cases of one instruction directory.

    python tools/fit_fp4_position.py <case dir> [tiles per half]
"""

import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.formats import E2M1, E2M3  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402

d_dir = pathlib.Path(sys.argv[1])
per_half = int(sys.argv[2]) if len(sys.argv) > 2 else 64
meta = json.loads((d_dir / "cases.json").read_text())
ins, a, b, c = gpu_cases.load(d_dir, meta)
d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
half, n = meta["scaled_from"], meta["random_tiles"]
ix = np.r_[0:per_half, half : half + per_half]
a, b, c, d = a[ix], b[ix], c[ix], d[ix]
base = tfdpa(meta["arch"], ins["types"], ins["shape"])
print(f"{meta['inst']}: {c.size} elements")
variants = [(f"e2m1 lsb {lsb}", dataclasses.replace(E2M1, lsb=lsb)) for lsb in range(5)]
variants.append(("byte read as e2m3 (bits 5:0)", E2M3))
for label, fmt in variants:
    p = dataclasses.replace(base, fa=fmt if ins["ta"] == "e2m1" else base.fa,
                            fb=fmt if ins["tb"] == "e2m1" else base.fb)
    bad = sum(dot(p, a[t, i].tolist(), b[t, j].tolist(), int(c[t, i, j])) != int(d[t, i, j])
              for t in range(len(ix)) for i in range(c.shape[1]) for j in range(c.shape[2]))
    print(json.dumps({"e2m1_as": label, "mismatches": bad}))
