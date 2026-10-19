"""Where the GPU and the model disagree: by tile index mod 4 (the m8n8k4 quad-pair), by
row and column, and a few cases decoded, to tell a harness bug from an arithmetic one.

    python tools/mismatch_map.py <case dir> [n cases to show]
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gpu_cases  # noqa: E402

d_dir = pathlib.Path(sys.argv[1])
show = int(sys.argv[2]) if len(sys.argv) > 2 else 5
meta = json.loads((d_dir / "cases.json").read_text())
ins, a, b, c = gpu_cases.load(d_dir, meta)
d = np.fromfile(d_dir / "d.bin", gpu_cases.WORD[ins["size"][ins["td"]]]).reshape(c.shape)
model = gpu_cases.model_output(meta["arch"], ins, a, b, c, jobs=8)
bad = d != model
print("mismatches", int(bad.sum()), "of", bad.size)
print("by tile % 4:", [int(bad[q::4].sum()) for q in range(4)])
print("by row:", bad.sum(axis=(0, 2)).tolist())
print("by col:", bad.sum(axis=(0, 1)).tolist())
half = meta["scaled_from"]
print("raw tiles with any mismatch:", int(bad[:half].any(axis=(1, 2)).sum()), "of", half)
print("scaled tiles with any mismatch:", int(bad[half:meta["random_tiles"]].any(axis=(1, 2)).sum()),
      "of", meta["random_tiles"] - half)
for t, i, j in np.argwhere(bad)[:show]:
    print(f"tile {t} ({i},{j}): a={[hex(x) for x in a[t, i]]} b={[hex(x) for x in b[t, j]]} c={hex(c[t, i, j])} "
          f"gpu={hex(d[t, i, j])} model={hex(model[t, i, j])}")
