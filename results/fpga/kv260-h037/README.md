# KV260 system for H-037: a 4 × 4 array of GST-FDPA units (MXFP4)

`rtl/tm_pl_array.v` with GST = 1, R = C = 4 and Q = 16, built from `97952fe` (not dirty). The
units are `rtl/gst_fdpa.sv` for `m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0`: K = L = 64, G = 16,
two UE8M0 scale factors per row and column per step, F = 35, XP = SP = 1. Jobs are
64 × 64 × 256.

| build | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| 4 × 4 MXFP4 | 58,460 (49.9 %) | 31,554 | 64 | 39 | −0.139 | **+0.527** |

- **The 16 units** take 42,621 LUT, about 2,660 each, and 4 DSP each. Out of context,
  `gst_fdpa` took 4,011 LUT (H-022).
- **The D**, in Verilator, is the 5070 Ti's, for MXFP4 and NVFP4 (`../../rtl/tm_pl_array-h037-97952fe.jsonl`,
  `../../gpu/h037-bs-gemm-97952fe`). NVFP4 would be its own build: four UE4M3 scale
  factors per step.
- **Board runs:** pending, at 187.5 MHz.

The images are not kept here; `sha256.txt` names them.
