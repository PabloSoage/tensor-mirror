"""Verilator -G flags that make rtl/tfdpa.sv one instruction of python/tensormirror/arch.py.

    python tools/rtl_params.py <arch> <D.A.B.C types> [shape]      e.g. Ampere f32.bf16.bf16.f32
prints the flags on one line, then L, the A and B word widths and the latency on a second
line, for tools/run_rtl_generic.sh.
"""

import sys

from tensormirror.arch import PTX_TYPES, tfdpa
from tensormirror.formats import FINITE, FN, IEEE, Format

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


def main() -> int:
    g, L, aw, bw = flags(*sys.argv[1:4])
    print(" ".join(g))
    print(L, aw, bw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
