# KV260 systems for H-035: RTX Blackwell arrays with KI < L

`rtl/tm_pl_array.v` as of `159f42a` (not dirty): double buffered (H-034), with KI, the k of
the emulated instruction. The RTX 5070 Ti's units, XP = SP = 1. Jobs are 64 × 64 × 256 and
the generics are in each `versions.txt`.

| build | unit | KI | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|---|---|
| 4 × 4 `f16.f16.f16.f16` | L = 16, F = 25 | 8 | 65,707 (56.1 %) | 50,510 | 256 | 39 | −0.110 | **+0.556** |
| 2 × 4 `f32.e4m3.e4m3.f32` | L = 32, F = 25 | 16 | 49,347 (42.1 %) | 28,890 | 0 | 31 | −0.011 | **+0.655** |

- **The units, in context:**
  - **f16:** 54,432 LUT for 16, about 3,400 each, with 16 DSP.
  - **FP8:** 34,001 LUT for 8, about 4,250 each, with no DSP: its products are LUTs. Out of
    context it took 6,921 (H-018), so it shrinks here.
- **Neither meets 214.29 MHz**, so both run at 187.5 MHz on the board.
- **The D** of these configurations, in Verilator, is the 5070 Ti's
  (`../../gpu/h035-gemm-159f42a`).
- **Board runs:** pending; the board was off.

The images are not kept here; `sha256.txt` names them.
