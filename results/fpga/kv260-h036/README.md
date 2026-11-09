# KV260 system for H-036: a 4 × 4 array of RTX Blackwell FP8 units

`rtl/tm_pl_array.v` with R = C = 4, Q = 16 and KI = 16, built from `73fef3f`. Its RTL is
that of `159f42a` (H-035); the commits between them only add evidence. Units are
`f32.e4m3.e4m3.f32` (L = 32, F = 25), XP = SP = 1, with jobs of 64 × 64 × 256.

| build | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| 4 × 4 `f32.e4m3.e4m3.f32`, KI = 16 | 81,743 (69.8 %) | 44,872 | 0 | 39 | −0.232 | **+0.434** |

- **The 16 units** take 65,959 LUT, about 4,120 each, and no DSP.
- **The D**, in Verilator, is the 5070 Ti's: `../../rtl/tm_pl_array-4x4-h036-159f42a.jsonl`,
  against the loops of `../../gpu/h035-gemm-159f42a`.
- **Board runs:** pending, at 187.5 MHz.

The images are not kept here; `sha256.txt` names them.
