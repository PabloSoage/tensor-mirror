# KV260 system for H-038: the 4 × 4 GST array for NVFP4

As `../kv260-h037`, for `m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3`: four UE4M3 scale
factors per row and column per step (NS = 4, UE4M3 = 1). Built from `4062ff1`, whose RTL is
that of `97952fe`; the commits between them only add evidence and a mutant run.

| build | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| 4 × 4 NVFP4 | 59,985 (51.2 %) | 32,083 | 64 | 39 | −0.187 | **+0.479** |

- **Against MXFP4:** 1,525 LUT more, for the wider scale memories and the UE4M3 scales.
- **The D**, in Verilator, is the 5070 Ti's (`../../rtl/tm_pl_array-h037-97952fe.jsonl`).
- **Board runs:** pending, at 187.5 MHz.

The images are not kept here; `sha256.txt` names them.
