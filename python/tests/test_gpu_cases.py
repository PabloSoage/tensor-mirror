import argparse
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "tools"))
import gpu_cases  # noqa: E402

from tensormirror.arch import tfdpa  # noqa: E402
from tensormirror.tfdpa import dot  # noqa: E402


def fake_gpu(out: pathlib.Path, variant: str, flip: bool) -> None:
    """Write d.bin as the model computes it, optionally with one bit flipped."""
    meta = json.loads((out / "cases.json").read_text())
    td, _, _, tc = variant.split(".")
    nt = meta["tiles"]
    a = np.fromfile(out / "a.bin", np.uint16).reshape(nt, 8, 4)
    b = np.fromfile(out / "b.bin", np.uint16).reshape(nt, 8, 4)
    c = np.fromfile(out / "c.bin", gpu_cases.WORD[tc]).reshape(nt, 8, 8)
    p = tfdpa("Volta", variant)
    d = np.array([[[dot(p, a[t, i].tolist(), b[t, j].tolist(), int(c[t, i, j])) for j in range(8)]
                   for i in range(8)] for t in range(nt)], dtype=gpu_cases.WORD[td])
    if flip:
        d[0, 3, 5] ^= 1
    d.tofile(out / "d.bin")


@pytest.mark.parametrize("variant", ["f32.f16.f16.f32", "f32.f16.f16.f16", "f16.f16.f16.f16"])
@pytest.mark.parametrize("flip", [False, True])
def test_check_catches_one_flipped_bit(tmp_path, variant, flip):
    gpu_cases.gen(argparse.Namespace(variant=variant, tiles=3, seed=7, out=tmp_path))
    fake_gpu(tmp_path, variant, flip)
    assert gpu_cases.check(argparse.Namespace(dir=tmp_path, mmasim=False)) == int(flip)
    report = json.loads((tmp_path / "check.json").read_text())
    assert report["gpu_vs_model"] == int(flip)
    assert all(v["rest_of_tile_ok"] for v in report["directed"].values())
