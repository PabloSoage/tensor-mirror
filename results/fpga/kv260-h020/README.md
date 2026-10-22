# KV260 systems for H-020: tm_pl_x as one instruction per GPU generation

`fpga/build_kv260.tcl` with `top tm_pl_x`, the generics of `tools/rtl_params.py` for the
instruction and `XP:1 SP:1` (`versions.txt` in each directory). The builds ran one after
another from `a0a176b` to `20faf52`, none dirty; `rtl/tfdpa.sv`, `rtl/tm_pl_x.v`,
`rtl/tm_regs.v`, `rtl/pipe_*.sv` and `fpga/build_kv260.tcl` are the same in all of them.
pl_clk0 asked at 200 MHz; on the board it is IOPLL / 7 = 214.29 MHz (`sta_div7.txt`,
`fpga/sta_rates.tcl`); `sta_div8.txt` is 187.5 MHz.

| GPU | instruction | L | LUT | FF | DSP | WNS at 214.29 MHz (ns) |
|---|---|---|---|---|---|---|
| A100 | m16n8k8.f32.f16.f16.f32 | 8 | 7,661 | 11,711 | 8 | +0.405 |
| A100 | m16n8k8.f32.bf16.bf16.f32 | 8 | 8,254 | 11,929 | 0 | +0.545 |
| A100 | m16n8k8.f16.f16.f16.f16 | 8 | 7,684 | 11,606 | 8 | +0.312 |
| A100 | m16n8k4.f32.tf32.tf32.f32 | 4 | 6,856 | 10,701 | 4 | +0.475 |
| RTX 5070 Ti | m16n8k32.f32.e4m3.e4m3.f32 | 32 | 12,510 | 14,002 | 0 | +0.343 |

LUT, FF and DSP are the whole system (PS, DMA, interconnect and tm). The images are not kept
here; `sha256.txt` names them.
