"""The integer block-scale model with MMA-Sim's published parameters against MMA-Sim itself.

    python tools/xcheck_bs.py <case dir>...      (dirs from tools/bs_cases.py gen)

Prints one JSON line per instruction that MMA-Sim models; every count should be 0.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bs_cases import load, mmasim_name, mmasim_output, model_output  # noqa: E402


def main() -> int:
    bad = 0
    for arg in sys.argv[1:]:
        meta, ins, x = load(pathlib.Path(arg))
        s = mmasim_output(ins, x)
        if s is None:
            continue
        model = model_output(meta["inst"], x, jobs=8, measured=False)
        n = int((model != s).sum())
        bad += n
        print(json.dumps({"inst": meta["inst"], "mmasim": mmasim_name(ins), "elements": int(s.size),
                          "model_vs_mmasim": n, "first": np.argwhere(model != s)[:3].tolist()}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
