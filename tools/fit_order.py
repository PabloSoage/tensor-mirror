"""Test non-fused summation orders against a GPU's m8n8k4 output (f16 operands, C and D
in f32): every product is exact in f32 (11 x 11 significand bits), and the candidates
add them and C with f32 round-to-nearest-even additions in different orders. NumPy's
float32 addition is IEEE RNE with subnormals, which is what each candidate assumes.

    python tools/fit_order.py <case dir of m8n8k4.f32.f16.f16.f32>
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

d_dir = pathlib.Path(sys.argv[1])
meta = json.loads((d_dir / "cases.json").read_text())
ins, a, b, c = gpu_cases.load(d_dir, meta)
assert ins["types"] == "f32.f16.f16.f32"
d = np.fromfile(d_dir / "d.bin", np.uint32).reshape(c.shape)
A = a.view(np.float16).astype(np.float32)  # (t, 8, 4)
B = b.view(np.float16).astype(np.float32)  # (t, 8, 4): B^T
C = c.view(np.float32)
with np.errstate(all="ignore"):
    P = A[:, :, None, :] * B[:, None, :, :]  # (t, 8, 8, 4), exact
    p0, p1, p2, p3 = (P[..., k] for k in range(4))
    cands = {
        "((p0+p1)+p2)+p3, then +c": (((p0 + p1) + p2) + p3) + C,
        "(p0+p1)+(p2+p3), then +c": ((p0 + p1) + (p2 + p3)) + C,
        "c+(p0+p1) then +(p2+p3)": (C + (p0 + p1)) + (p2 + p3),
        "((c+p0)+p1)+p2)+p3": (((C + p0) + p1) + p2) + p3,
        "c+((p0+p1)+(p2+p3))": C + ((p0 + p1) + (p2 + p3)),
        "(p0+p1)+((p2+p3)+c)": (p0 + p1) + ((p2 + p3) + C),
        "((p3+p2)+p1)+p0, then +c": (((p3 + p2) + p1) + p0) + C,
    }
finite = np.isfinite(d.view(np.float32))
print(f"{d.size} elements, {int(finite.sum())} finite on the GPU")
for name, r in cands.items():
    r = r.astype(np.float32)
    same = r.view(np.uint32) == d
    same |= (r == 0) & (d.view(np.float32) == 0)  # +0 / -0
    print(f"{name:32s} mismatches on finite: {int((~same & finite).sum())}")
