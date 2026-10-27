"""T-FDPA parameters per NVIDIA architecture and ``mma`` type.

``L_max``, ``F`` and the rounding are MMA-Sim's Table 4 (arXiv 2511.10909); ``e_zero``
is not in the paper and comes from MMA-Sim's code (``nv_ptx/sim.py``, commit
54294d4). They are the published values, to be checked on our own GPUs before the RTL
relies on them.
"""

from .formats import BF16, E2M1, E2M3, E3M2, E4M3, E5M2, E8M13, FP16, FP32, TF32, Format
from .tfdpa import TFDPA

F_BITS = {"Volta": 23, "Turing": 24, "Ampere": 24, "Ada Lovelace": 24,
          "Hopper": 25, "Blackwell": 25, "RTX Blackwell": 25}
L_MAX_BYTES = {"Volta": 8, "Turing": 16, "Ampere": 16, "Ada Lovelace": 16,
               "Hopper": 32, "Blackwell": 32, "RTX Blackwell": 32}
# PTX type -> (format, bytes per element in registers)
PTX_TYPES = {"f32": (FP32, 4), "tf32": (TF32, 4), "f16": (FP16, 2), "bf16": (BF16, 2),
             "e5m2": (E5M2, 1), "e4m3": (E4M3, 1), "e3m2": (E3M2, 1), "e2m3": (E2M3, 1), "e2m1": (E2M1, 1)}
FP8 = {"e5m2", "e4m3"}
# (C in f16, C in f32); for f32 it is -108 - F [CIT MMA-Sim §4.2]
E_ZERO = {"Volta": (-20, -131), "Turing": (-21, -132), "Ampere": (-21, -132), "Ada Lovelace": (-21, -132),
          "Hopper": (-21, -133), "Blackwell": (-21, -133), "RTX Blackwell": (-21, -133)}


def e_zero(arch: str, fc: Format) -> int:
    return E_ZERO[arch][0 if fc is FP16 else 1]


def tfdpa(arch: str, types: str, shape: str = "", measured: bool = True) -> TFDPA:
    """``types`` in PTX order, D.A.B.C, e.g. ``"f32.f16.f16.f32"``; ``shape`` as in PTX.

    ``measured = False`` gives MMA-Sim's published parameters; by default the departures we
    measured on GPUs apply, each marked [MED] below.
    """
    d, a, b, c = types.split(".")
    (fa, size), (fb, _), (fc, _), (fd, _) = PTX_TYPES[a], PTX_TYPES[b], PTX_TYPES[c], PTX_TYPES[d]
    F = F_BITS[arch]
    ez = e_zero(arch, fc)
    if arch == "Ada Lovelace" and a in FP8:  # 13-bit FP8 accumulation [CIT MMA-Sim Table 4]
        F = 13
        if fd is FP32:
            fd = E8M13
    if measured and arch in ("Turing", "Ampere", "Ada Lovelace") and fd is FP16:
        # With D in f16 only e_zero = -20 reproduces tools/probe_f16_round.py, not the -21
        # of MMA-Sim 54294d4. [MED] on an A100 and an RTX 2060 Super, 2026-10-27.
        # MMA-Sim v2.1.1 has -20 for all three; Ada is that value [CIT], not measured.
        ez = -20
    if measured and arch == "Turing" and shape == "m8n8k4":
        # Turing runs Volta's m8n8k4 with Volta's 23-bit sum, not its own 24 (MMA-Sim
        # assumes 24 and has no Turing kernel for it). A zero C in f16 does not set the
        # alignment: every e_zero <= -25 matches all the product exponents probed (-14 to
        # -28, tools/probe_ezero.py), and they all give the same bits for f16 operands, whose
        # smallest product exponent is -28. [MED] on an RTX 2060, 2026-10-19
        # With D in f16 the sum stops at 2**-42, as on the V100 below: only -19 matches
        # (tools/probe_f16_round.py). [MED] on an RTX 2060, 2026-10-21
        F = 23
        if fc is FP16:
            ez = -19 if fd is FP16 else -25
    if measured and arch == "Volta" and fc is FP16:
        # MMA-Sim has -20 for both outputs; the V100 differs by D. With D in f32 a zero C in
        # f16 drops no product bits (every e_zero <= -25 matches tools/probe_ezero.py); with
        # D in f16 the sum stops at 2**-42, which only e_zero = -19 reproduces
        # (tools/probe_f16_round.py). [MED] on four V100-SXM2, 2026-10-21
        ez = -19 if fd is FP16 else -25
    if measured:
        # kind::f8f6f4 reads an E2M1 byte as E2M3 in bits 5:0: bits 1:0 are two more
        # mantissa bits, not padding (MMA-Sim drops them). With them zero it is E2M1 in
        # bits 5:2. [MED] on an RTX 5070 Ti, 2026-10-19
        fa = E2M3 if a == "e2m1" else fa
        fb = E2M3 if b == "e2m1" else fb
    return TFDPA(fa=fa, fb=fb, fc=fc, fd=fd, L_max=L_MAX_BYTES[arch] // size, F=F,
                 rounding="rne" if fd is FP16 else "rz", e_zero=ez)
