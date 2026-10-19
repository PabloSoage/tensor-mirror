"""T-FDPA parameters per NVIDIA architecture and ``mma`` type.

``L_max``, ``F`` and the rounding are MMA-Sim's Table 4 (arXiv 2511.10909); ``e_zero``
is not in the paper and comes from MMA-Sim's code (``nv_ptx/sim.py``, commit
54294d4). They are the published values, to be checked on our own GPUs before the RTL
relies on them.
"""

from .formats import BF16, FP16, FP32, Format
from .tfdpa import TFDPA

F_BITS = {"Volta": 23, "Turing": 24, "Ampere": 24, "Ada Lovelace": 24,
          "Hopper": 25, "Blackwell": 25, "RTX Blackwell": 25}
L_MAX_BYTES = {"Volta": 8, "Turing": 16, "Ampere": 16, "Ada Lovelace": 16,
               "Hopper": 32, "Blackwell": 32, "RTX Blackwell": 32}
FORMATS = {"f16": FP16, "bf16": BF16, "f32": FP32}
# (C in f16, C in f32); for f32 it is -108 - F [CIT MMA-Sim §4.2]
E_ZERO = {"Volta": (-20, -131), "Turing": (-21, -132), "Ampere": (-21, -132), "Ada Lovelace": (-21, -132),
          "Hopper": (-21, -133), "Blackwell": (-21, -133), "RTX Blackwell": (-21, -133)}


def e_zero(arch: str, fc: Format) -> int:
    return E_ZERO[arch][0 if fc is FP16 else 1]


def tfdpa(arch: str, types: str) -> TFDPA:
    """``types`` in PTX order, D.A.B.C, e.g. ``"f32.f16.f16.f32"``."""
    d, a, b, c = (FORMATS[t] for t in types.split("."))
    return TFDPA(
        fa=a,
        fb=b,
        fc=c,
        fd=d,
        L_max=L_MAX_BYTES[arch] // (a.width // 8),
        F=F_BITS[arch],
        rounding="rne" if d is FP16 else "rz",
        e_zero=e_zero(arch, c),
    )
