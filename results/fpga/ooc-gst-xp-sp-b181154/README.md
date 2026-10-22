# Pipelined GST-FDPA out of context on the XCK26 (H-022)

`fpga/synth_tfdpa.tcl` with `TOP gst_fdpa XP 1 SP 1`: `rtl/gst_fdpa.sv` at `b181154`, each
group's sum and the sum of the terms as pipelined adder trees, stage 6 split; latency 12.
Placed and routed out of context at 4.0 ns with Vivado 2026.1, the flow of H-021.

| configuration | LUT | FF | DSP | WNS (ns) | critical path | WNS without XP, SP (ns) |
|---|---|---|---|---|---|---|
| NB = 2, UE8M0 | 4,011 | 2,277 | 4 | +0.410 | stage 2: scale exponent to the term shifts | −3.044 |
| NB = 4, UE4M3 | 4,128 | 2,360 | 4 | +0.399 | stage 2: scale significand to the term shifts | −2.945 |
| NB = 4, UE8M0 | 4,040 | 2,369 | 4 | +0.377 | stage 2: the DSP product to `mag2` | −2.938 |

Each directory holds the routed design's utilization, timing summary and critical path, and
`summary.txt` with the generics and the WNS.
