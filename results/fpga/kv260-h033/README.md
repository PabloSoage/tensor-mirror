# KV260 systems for H-033: an 8 x 4 array of Ampere f16 units

`rtl/tm_pl_array.v` as in H-032, with R = 8, C = 4, Q = 16 and one job of M x N x K =
128 x 64 x 256. Built from `4c990e9` (the RTL of `5e29cd9`), not dirty. The A100's units,
XP = SP = 1. The generics are in each `versions.txt`.

| types | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| f16 | 88,562 (75.6 %) | 61,037 | 256 | 55 | **−0.323** | +0.343 |
| f32 | 91,734 (78.3 %) | 62,833 | 256 | 55 | **−0.290** | +0.376 |

- **The 32 units** (`u_array`) take 77,198 LUT in f16, 2,412 each, and 76,179 in f32.
- **`tm_pl_array` outside the units:** 5,709 LUT in f16, of which 4,096 are the 32 C/D
  banks in LUTRAM. In f32 the banks take 8,192.
- **BRAM:** A takes 8 banks of 4 BRAM36, B 4 banks of 4, and the DMA the rest.
- **Neither closes at 214.29 MHz**, by about 0.3 ns, so on the board both run at 187.5 MHz.
- **On the board** both give the RTX 2060 Super's D byte for byte
  (`../kv260-board-2026-10-28/x16`, `x32`).

The images are not kept here; `sha256.txt` names them.
