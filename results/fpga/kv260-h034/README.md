# KV260 systems for H-034: the double-buffered array

`rtl/tm_pl_array.v` with two sides per bank (`1ddeb6d`, not dirty): the load, the array and
the drain overlap. The A100's units, XP = SP = 1. 4 × 4 runs jobs of 64 × 64 × 256 and 8 × 4
runs jobs of 128 × 64 × 256. The generics are in each `versions.txt`.

| build | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| 4 × 4 f16 | 50,696 (43.3 %) | 37,727 | 128 | 39 | **+0.014** | +0.680 |
| 4 × 4 f32 | 54,618 (46.6 %) | 38,667 | 128 | 39 | **−0.024** | +0.642 |
| 8 × 4 f16 | 96,771 (82.6 %) | 63,418 | 256 | 55 | −0.522 | **+0.144** |
| 8 × 4 f32 | 104,835 (89.5 %) | 65,263 | 256 | 55 | −0.657 | **+0.009** |

- **Against the single-buffered builds** (`../kv260-h032`, `../kv260-h033`):
  - BRAM and DSP are the same. A and B take the second job in the BRAMs' unused depth.
  - LUT grow by 3,341 (4 × 4 f16), 5,760 (4 × 4 f32), 8,209 (8 × 4 f16) and 13,101
    (8 × 4 f32), mostly from the second C/D LUTRAM. In f32 those banks are twice as wide.
- **4 × 4 f32 misses 214.29 MHz by 24 ps**, so on the board it would run at 187.5 MHz.
- **The cycles**, from Verilator, are in `../../rtl/tm_pl_array-h034-1ddeb6d.jsonl`.
- **Board runs:** pending; the board was off.

The images are not kept here; `sha256.txt` names them.
