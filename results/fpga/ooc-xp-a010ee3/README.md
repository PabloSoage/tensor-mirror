# One T-FDPA per generation with pipelined trees, out of context on the XCK26 (H-017)

`fpga/synth_tfdpa.tcl`: `rtl/tfdpa.sv` at `a010ee3` with `PIPE=1` and `XP=1`, so the maximum
exponent and the sum of the terms are binary trees with a register every 2 levels. Latency is
10 cycles for L = 4, 12 for L = 8 and 16, and 14 for L = 32. Placed and routed out of context at
4.0 ns (250 MHz) with Vivado 2026.1, same flow and parameters as `ooc-generations-e443189` (H-015).

| configuration | L | LUT | FF | DSP | WNS (ns) | critical path | H-015 WNS (ns) |
|---|---|---|---|---|---|---|---|
| Volta f32.f16.f16.f32 | 4 | 1,366 | 797 | 4 | −0.263 | stage 6 shift | +0.169 |
| Volta f16.f16.f16.f16 | 4 | 1,414 | 781 | 4 | +0.106 | stage 6 shift | +0.063 |
| Ampere f32.tf32.tf32.f32 | 4 | 1,482 | 853 | 4 | −0.057 | stage 6 shift | +0.134 |
| Ampere f32.f16.f16.f32 | 8 | 2,148 | 1,575 | 8 | +0.294 | stage 6 shift | −0.883 |
| Ampere f32.bf16.bf16.f32 | 8 | 2,745 | 1,792 | 0 | +0.233 | stage 6 shift | −0.892 |
| Ada f32.e4m3.e5m2.f32 | 16 | 2,353 | 1,832 | 0 | +0.142 | stage 6 shift | −5.779 |
| Hopper f32.f16.f16.f32 | 16 | 3,828 | 2,837 | 16 | +0.097 | stage 6 shift | −4.740 |
| RTX Blackwell f32.e2m1.e3m2.f32 | 32 | 6,272 | 3,389 | 0 | +0.153 | max tree to stage 2 shifts | −15.474 |
| RTX Blackwell f32.e4m3.e4m3.f32 | 32 | 6,946 | 3,605 | 0 | +0.013 | stage 6 shift | −15.019 |

Stage 2 is no longer critical from L = 8 up. The critical path of every configuration but one is
now stage 6 (`thr5`/`msb5` to `q6`, the normalising shift), which `XP` does not change. In the
two L = 4 configurations that fail, that path has the same logic delay as in H-015 (1.81 vs 1.88 ns
for Volta f32.f16.f16.f32) and 0.5 ns more routing (2.45 vs 1.95 ns): stage 6 is at the edge of
4 ns, and the placement of the extra registers decides the sign. In RTX Blackwell
f32.e2m1.e3m2.f32 the path is the maximum exponent fanning out to the 33 alignment shifters, 83 %
routing.

Each directory holds the routed design's utilization, timing summary and critical path, and
`summary.txt` with the generics and the WNS.
