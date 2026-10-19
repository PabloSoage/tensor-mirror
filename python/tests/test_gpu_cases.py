import argparse
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "tools"))
import gpu_cases  # noqa: E402


def fake_gpu(out: pathlib.Path, flip: bool) -> None:
    """Write d.bin as the model computes it, optionally with one bit flipped."""
    meta = json.loads((out / "cases.json").read_text())
    ins, a, b, c = gpu_cases.load(out, meta)
    d = gpu_cases.model_output(meta["arch"], ins, a, b, c, jobs=1)
    if flip:
        d[0, 3, 5] ^= 1
    d.tofile(out / "d.bin")


@pytest.mark.parametrize(("arch", "inst"), [
    ("Volta", "m8n8k4.f32.f16.f16.f32"), ("Volta", "m8n8k4.f16.f16.f16.f16"),
    ("Turing", "m16n8k8.f32.f16.f16.f32"), ("Ampere", "m16n8k8.f32.tf32.tf32.f32"),
    ("RTX Blackwell", "m16n8k32.f8f6f4.f32.e2m1.e3m2.f32"), ("Ada Lovelace", "m16n8k16.f16.e4m3.e5m2.f16"),
])
@pytest.mark.parametrize("flip", [False, True])
def test_check_catches_one_flipped_bit(tmp_path, arch, inst, flip):
    gpu_cases.gen(argparse.Namespace(arch=arch, inst=inst, tiles=4, seed=7, out=tmp_path))
    fake_gpu(tmp_path, flip)
    assert gpu_cases.check(argparse.Namespace(dir=tmp_path, mmasim=False, jobs=1)) == int(flip)
    report = json.loads((tmp_path / "check.json").read_text())
    assert report["gpu_vs_model"] == int(flip)
    assert all(v["rest_of_tile_ok"] for v in report["directed"].values())


def test_scaled_words_stay_near_one():
    import random

    from tensormirror.arch import PTX_TYPES
    from tensormirror.formats import decode

    rng = random.Random(3)
    for t in ("f16", "bf16", "tf32", "e4m3", "e5m2", "e3m2", "e2m3", "e2m1"):
        fmt, size = PTX_TYPES[t]
        for _ in range(200):
            x = decode(fmt, gpu_cases.scaled_word(rng, fmt, size, 2))
            assert x.kind == "num" and x.sig > 0 and abs(x.exp) <= 2, (t, x)
    assert np.uint8  # numpy is a dependency of the tool
