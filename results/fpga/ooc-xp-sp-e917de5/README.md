# One T-FDPA per generation with pipelined trees and a split stage 6, out of context (H-018)

`fpga/synth_tfdpa.tcl`: `rtl/tfdpa.sv` at `e917de5` with `PIPE=1`, `XP=1` and `SP=1`: the
trees of `ooc-xp-a010ee3` (H-017) and a register in stage 6 between the shift amount and the
shift. Latency is 11 cycles for L = 4, 13 for L = 8 and 16, and 15 for L = 32. Placed and
routed out of context at 4.0 ns (250 MHz) with Vivado 2026.1, same flow and parameters as
H-015 and H-017.

| configuration | L | LUT | FF | WNS (ns) | critical path | WNS with XP only (ns) |
|---|---|---|---|---|---|---|
| Volta f32.f16.f16.f32 | 4 | 1,296 | 847 | +0.510 | max tree, first level | −0.263 |
| Volta f16.f16.f16.f16 | 4 | 1,332 | 823 | +0.097 | stage 7 rounding (`half6` to `d`) | +0.106 |
| Ampere f32.tf32.tf32.f32 | 4 | 1,370 | 905 | +0.493 | adder tree, first level | −0.057 |
| Ampere f32.f16.f16.f32 | 8 | 2,078 | 1,627 | +0.429 | adder tree, first level | +0.294 |
| Ampere f32.bf16.bf16.f32 | 8 | 2,667 | 1,845 | +0.703 | adder tree, first level | +0.233 |
| Ada f32.e4m3.e5m2.f32 | 16 | 2,317 | 1,874 | +0.791 | max tree, first level | +0.142 |
| Hopper f32.f16.f16.f32 | 16 | 3,661 | 2,889 | +0.286 | max tree to stage 2 shifts | +0.097 |
| RTX Blackwell f32.e2m1.e3m2.f32 | 32 | 6,244 | 3,444 | +0.205 | max tree to stage 2 shifts | +0.153 |
| RTX Blackwell f32.e4m3.e4m3.f32 | 32 | 6,921 | 3,660 | +0.421 | adder tree, first level | +0.013 |

All nine meet 4.0 ns. The split stage 6 is no longer critical anywhere. The paths that remain:
- the first level of the max or adder tree (2 levels between registers);
- the fan-out of the maximum exponent to the alignment shifters (Hopper, RTX Blackwell e2m1,
  over 80 % routing);
- with an f16 output, the rounding of stage 7.

Each directory holds the routed design's utilization, timing summary and critical path, and
`summary.txt` with the generics and the WNS.
