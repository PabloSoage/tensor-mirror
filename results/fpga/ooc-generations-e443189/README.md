# One T-FDPA per generation, out of context on the XCK26 (H-015)

`fpga/synth_tfdpa.tcl`: `rtl/tfdpa.sv` with `PIPE=1`, so inputs and output are registered
(8 stages). Placed and routed out of context at 4.0 ns (250 MHz) with Vivado 2026.1. The
parameters of each instruction come from `tools/rtl_params.py`.

The runs started with the RTL of `b36519b`. `rtl/tfdpa.sv` gained the `ST` parameter
(`e443189`) while they ran, and Vivado reads the file at the start of each run. With `ST = 0`
the added logic is constant and the scale inputs are unused, so the netlist is the same.

| configuration | L | F | LUT | FF | DSP | WNS (ns) | critical path |
|---|---|---|---|---|---|---|---|
| Volta f32.f16.f16.f32 | 4 | 23 | 1,208 | 621 | 4 | +0.169 | stage 6 shift |
| Volta f16.f16.f16.f16 | 4 | 23 | 1,273 | 602 | 4 | +0.063 | stage 6 shift |
| Ampere f32.tf32.tf32.f32 | 4 | 24 | 1,307 | 664 | 4 | +0.134 | stage 6 shift |
| Ampere f32.f16.f16.f32 | 8 | 24 | 1,817 | 961 | 8 | −0.883 | stage 2: max exponent, shifts |
| Ampere f32.bf16.bf16.f32 | 8 | 24 | 2,394 | 1,233 | 0 | −0.892 | stage 2 |
| Ada f32.e4m3.e5m2.f32 | 16 | 13 | 2,202 | 1,295 | 0 | −5.779 | stage 2 |
| Hopper f32.f16.f16.f32 | 16 | 25 | 3,218 | 1,612 | 16 | −4.740 | stage 2 |
| RTX Blackwell f32.e2m1.e3m2.f32 | 32 | 25 | 4,910 | 2,614 | 0 | −15.474 | stage 2 |
| RTX Blackwell f32.e4m3.e4m3.f32 | 32 | 25 | 5,554 | 2,865 | 0 | −15.019 | stage 2 |

From L = 8 up, the critical path is stage 2, from `ex1` to `sh2`: the maximum of L + 1
exponents, then each term's shift. The maximum is a linear chain, so its delay grows with L.
A comparator tree with registers inside would bring it back under 4 ns.
