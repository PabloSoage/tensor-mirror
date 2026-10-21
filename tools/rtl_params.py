"""Verilator -G flags that make rtl/tfdpa.sv one instruction of python/tensormirror/arch.py.

    python tools/rtl_params.py <arch> <D.A.B.C types> [shape]      e.g. Ampere f32.bf16.bf16.f32
    python tools/rtl_params.py bs <block_scale instruction>         kind::mxf8f6f4 only (ST = 1)
prints the flags on one line, then L, the A and B word widths and the latency on a second
line, for tools/run_rtl_generic.sh.
"""

import sys

from tensormirror.arch import PTX_TYPES, tfdpa
from tensormirror.blockscale import block_scale
from tensormirror.formats import FINITE, FN, FP32, IEEE, Format

SPEC = {IEEE: 0, FN: 1, FINITE: 2}


def fmt_flags(x: str, f: Format, word: int | None = None) -> list[str]:
    flags = [f"-G{x}E={f.ebits}", f"-G{x}M={f.mbits}", f"-G{x}LSB={f.lsb}"]
    if word is not None:
        flags += [f"-G{x}W={word}", f"-G{x}SPEC={SPEC[f.specials]}"]
    return flags


def flags(arch: str, types: str, shape: str = "") -> tuple[list[str], int, int, int]:
    p = tfdpa(arch, types, shape)
    _, a, b, _ = types.split(".")
    aw, bw = PTX_TYPES[a][1] * 8, PTX_TYPES[b][1] * 8
    g = [f"-GL={p.L_max}", f"-GF={p.F}", f"-GEZERO={p.e_zero}", f"-GRNE={int(p.rounding == 'rne')}"]
    g += fmt_flags("A", p.fa, aw) + fmt_flags("B", p.fb, bw) + fmt_flags("C", p.fc) + fmt_flags("D", p.fd)
    return g, p.L_max, aw, bw


def bs_flags(inst: str) -> tuple[list[str], int, int, int]:
    """ST-FDPA: one group of K terms, one scale pair, f32 c and d rounded toward zero."""
    p = block_scale(inst)
    assert p.kind == "st", "GST instructions are rtl/gst_fdpa.sv"
    g = [f"-GL={p.K}", f"-GF={p.F}", f"-GEZERO={p.e_zero}", "-GRNE=0", "-GST=1"]
    g += fmt_flags("A", p.fa, 8) + fmt_flags("B", p.fb, 8) + fmt_flags("C", FP32) + fmt_flags("D", FP32)
    return g, p.K, 8, 8


def main() -> int:
    g, L, aw, bw = bs_flags(sys.argv[2]) if sys.argv[1] == "bs" else flags(*sys.argv[1:4])
    print(" ".join(g))
    print(L, aw, bw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
