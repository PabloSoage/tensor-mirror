# GST-FDPA out of context on the XCK26 (H-021)

`fpga/synth_tfdpa.tcl` with `TOP gst_fdpa`: `rtl/gst_fdpa.sv` (unchanged since `e443189`) with
`PIPE=1` (8 stages, no pipelined trees), K = 64, groups of 16, F = 35. Placed and routed out
of context at 4.0 ns with Vivado 2026.1, the flow of H-015.

| configuration | instructions | LUT | FF | DSP | WNS (ns) | critical path |
|---|---|---|---|---|---|---|
| NB = 2, UE8M0 | mxf4.2x, mxf4nvf4.2x | 3,973 | 1,878 | 4 | −3.044 | stage 2: group sum times scale, 21 levels |
| NB = 4, UE4M3 | mxf4nvf4.4x ue4m3 (NVFP4) | 4,070 | 1,930 | 4 | −2.945 | stage 2, 20 levels |
| NB = 4, UE8M0 | mxf4nvf4.4x ue8m0 | 3,998 | 1,930 | 4 | −2.938 | stage 2, 21 levels |

The sum of a group's 16 products is a chain, as written; Vivado does not rebalance it.

Each directory holds the routed design's utilization, timing summary and critical path, and
`summary.txt` with the generics and the WNS.
