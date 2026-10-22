# KV260 systems for H-023: tm_pl_x with the GST-FDPA (MXFP4, NVFP4)

`fpga/build_kv260.tcl` with `top tm_pl_x`, `GST:1` and the generics of
`tools/rtl_params.py gst` for the instruction, with `XP:1 SP:1` (`versions.txt` in each
directory). The three builds ran one after another from `11f5b3f` to `6bee9b2`, none dirty;
the RTL and `fpga/` are the same in all of them. pl_clk0 asked at 200 MHz; on the board it
is IOPLL / 7 = 214.29 MHz (`sta_div7.txt`, `fpga/sta_rates.tcl`); `sta_div8.txt` is 187.5 MHz.

| GPU | instruction | scales | LUT | FF | DSP | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|---|
| RTX 5070 Ti | m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0 | 2 × ue8m0 | 9,573 | 12,616 | 4 | +0.778 | +1.444 |
| RTX 5070 Ti | m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0 | 2 × ue8m0 | 9,572 | 12,616 | 4 | +0.696 | +1.362 |
| RTX 5070 Ti | m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3 | 4 × ue4m3 | 9,683 | 12,696 | 4 | +0.582 | +1.248 |

The first two instructions have the same generics, so the same unit; the systems differ in
the `BUILD` register and in place and route. LUT, FF and DSP are the whole system (PS, DMA, interconnect and tm). The
images are not kept here; `sha256.txt` names them. Board runs:
`../kv260-board-2026-10-22/` (g1 to g3).
